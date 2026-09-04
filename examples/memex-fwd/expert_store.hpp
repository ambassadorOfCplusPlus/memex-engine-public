// REZIDENTNOE HRANILISHCHE EKSPERTOV V OZU (shag 4 plana).
//
// ZACHEM ONO SUSHCHESTVUET, i eto ne "chtoby bylo bystree".
//
// Coder Next IQ3_XXS: 48 sloev x 512 ekspertov, ekspert 1,03 MiB, vsego 24,6 GiB tolko
// ekspertov. Na mashine 31,9 GiB fizicheskoj pamjati, iz nih svobodno okolo 23,5 - to est
// eksperty v strannichnyj kesh NE VLEZAJUT, i eto izmereno, a ne predpolozheno: --gen 64 dal
// 254 ms/token protiv 131 ms na --gen 8, potomu chto dlinnaja generacija vytesnjaet iz kesha
// rovno to, chto sledujushchij token snova prochitaet. Kazhdyj promah - 2,74 ms s SSD
// (bench/io_miss_cost.py, mimo kesha, NO_BUFFERING), i pri desjatkah promahov na token eto
// sotni millisekund.
//
// Strannichnyj kesh nelzja poprosit derzhat imenno tot podnabor, kotoryj nuzhen. On vytesnjaet
// po svoej politike i ne znaet, chto obrashchenija k ekspertam raspredeleny ochen koso: srez
// iz 410 samyh chastyh ekspertov na sloj pokryvaet 99,67% obrashchenij na sledujushchij token
// (bench/route_lab.py, zatravka s chuzhih tekstov). Hranilishche - eto sposob nazvat etot
// podnabor javno i zakrepit ego v PRIVATNOJ pamjati, kotoruju nikto ne vytesnit.
//
// Vtoraja prichina, ne menee vazhnaja: PRIVATNAJA pamjat pozvoljaet REPAK v _R4. Otobrazhenie
// fajla tolko dlja chtenija (CreateFileMappingA(PAGE_READONLY)), i perepisat v njom nichego
// nelzja; a repack IQ3_XXS -> IQ3_XXS_R4 po zameram avtora forka stoit 1,96x na scheto
// ekspertov. Repack zdes pod otdelnym flagom, potomu chto on MENJAET BAJTY otveta
// (u _R4 drugoj vec_dot_type dlja aktivacii), i eto nado nazyvat, a ne prjatat.
//
// USTROJSTVO. Na sloj - tri tenzora v porjadke SLOTOV, a ne ekspertov:
//
//     gate [n_embd, n_ff, S]   up [n_embd, n_ff, S]   down [n_ff, n_embd, S]
//
// gde S = C + Z: C rezidentnyh slotov (politika) i Z zapasnyh (sinhronnye promahi). Tipy i
// formy berutsja S TENZOROV MODELI, a ne zashity: u IQ3_XXS gate/up eto IQ2_S, a down
// IQ3_XXS, i u IQ4_XS oni drugie.
//
// POLITIKA - CHASTOTA S ZATRAVKOJ. Schjotchik kazhdogo eksperta na kazhdom sloe = zatravka s
// kalibrovochnogo korpusa (fajl --expert-prior, gotovitsja bench/expert_prior.py) plius
// vybory samogo dokumenta (prefill + sgenerirovannoe). Raz v `period` tokenov verhnie C po
// schjotchikam objavljajutsja rezidentnymi; sloty vytesnennyh zapolnjajutsja novymi. Bez
// zatravki plato popadanij 98,6% pri ljubom bjudzhete - eto PERVYE POJAVLENIJA eksperta v
// dokumente, i nikakaja istorija ih ne dostajot (bench/first_seen.py).
//
// PROMAH - REZHIM "ZHDAT", i drugogo rezhima zdes net. Esli ekspert nuzhen, a slota u nego
// net, on chitaetsja SINHRONNO v zapasnoj slot, i tolko posle etogo idjot mul_mat_id. Otvet
// modeli ot hranilishcha NE ZAVISIT (krome repaka, kotoryj nazvan otdelno). Rezhim "propustit
// eksperta" ne realizovan namerenno: on bystree i dajot drugoj otvet, a vydavat eto za
// uskorenie zdes zapreshcheno.
//
// PEREVOD id -> slot DELAETSJA V GRAFE, i eto edinstvennoe mesto, gde u etogo dizajna byl
// vybor. Ideal - perevesti na hoste do grafa, kak u GpuExperts, no zdes eto nevozmozhno:
// `sel` eto ggml_top_k VNUTRI grafa (a pri --gpu-static logity marshrutizatora prihodjat s
// karty tozhe vnutri grafa), tak chto do zapuska grafa vyborov ne znaet nikto. Poetomu mezhdu
// top-k i mul_mat_id stavitsja odin uzel ggml_map_custom1 s n_tasks=1 - tot zhe prijom, chto
// uzhe nesjot ves kartochnyj sloj (gpu_static.cpp: ggml_map_custom3). Etot uzel i perevodit
// id v slot, i - pri promahe - chitaet eksperta v zapasnoj slot. Zamena celocheslennaja:
// nikakih bajt rezultata ona ne kasaetsja.
#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include "ggml.h"
#include "ggml-backend.h"

namespace memex {

// Pamjat PROCESSA, a ne mashiny: rabochij nabor i schjotchik promahov stranic. Nuzhny imenno
// oni - "svobodno v sisteme" ne otvechaet na vopros, skolko iz hranilishcha realno lezhit v
// OZU, a chislo promahov stranic za generaciju i est chislo chtenij s diska, kotorye dvizhok
// ne uvidel sam.
struct ProcMem {
    bool     ok     = false;
    uint64_t rss    = 0;   // WorkingSetSize
    uint64_t peak   = 0;   // PeakWorkingSetSize
    uint64_t faults = 0;   // PageFaultCount - MJAGKIE I ZHJOSTKIE VMESTE, sm. zamechanie nizhe
};
// PageFaultCount schitaet VSE promahi stranic, a ne tolko zhjostkie: Windows ne otdajot
// zhjostkie otdelno ni cherez GetProcessMemoryInfo, ni cherez QueryProcessCycleTime. Poetomu
// eto verhnjaja granica chtenij s diska, i tak ono i pechataetsja. Otdelnyj zhjostkij schjot
// est tolko u schjotchikov proizvoditelnosti procesa, i ego zdes NE snimaem.
ProcMem proc_mem();

struct ExpertStoreConfig {
    int n_layer   = 0;
    int n_expert  = 0;
    int n_used    = 0;    // top-k marshrutizatora
    int capacity  = 0;    // C: rezidentnyh slotov na sloj. 0 - hranilishche vykljucheno
    int spares    = 16;   // Z: zapasnyh slotov na sloj pod sinhronnye promahi
    int period    = 16;   // tokenov mezhdu perestanovkami nabora
    bool repack   = false;
    // Put k gguf-fajlu modeli. Esli zadan, sloty zapolnjajutsja PRJAMYM CHTENIEM fajla mimo
    // strannichnogo kesha (CreateFile FILE_FLAG_NO_BUFFERING), a ne kopirovaniem iz mmap. Eto
    // i est smysl shaga: hranilishche - EDINSTVENNAJA kopija rezidentnyh ekspertov, stranicy
    // mmap teh zhe ekspertov ne fol'tjatsja i ne konkurirujut za pamjat. Pustaja stroka -
    // staroe povedenie (memcpy iz mmap), ostavleno dlja sverki.
    std::string gguf_path;
};

struct ExpertStoreStats {
    uint64_t tokens       = 0;   // tokenov dekoda, proshedshih cherez hranilishche
    uint64_t picks        = 0;   // obrashchenij k MARSHRUTIZIRUEMYM ekspertam
    uint64_t hits         = 0;   // iz nih te, u kogo slot uzhe byl
    uint64_t sync_misses  = 0;   // te, kogo prishlos prochitat sinhronno
    uint64_t refreshes    = 0;
    uint64_t fills        = 0;   // kopirovanij eksperta v slot (vkljuchaja pervichnuju zalivku)
    uint64_t fill_bytes   = 0;
    uint64_t evictions    = 0;
    double   ms_sync      = 0.0; // vremja vnutri grafa na sinhronnyh promahah
    double   ms_fill      = 0.0; // vsjo vremja kopirovanij (zalivka + perestanovki + promahi)
    double   ms_repack    = 0.0;
    double   ms_map       = 0.0; // vremja samogo perevoda id -> slot (bez chtenij)

    double hit_rate() const { return picks ? double(hits) / double(picks) : 0.0; }
    double misses_per_token() const {
        return tokens ? double(sync_misses) / double(tokens) : 0.0;
    }
    ExpertStoreStats since(const ExpertStoreStats& b) const;
};

class ExpertStore {
  public:
    ~ExpertStore();

    // Skolko rezidentnyh slotov na sloj vlezaet v `avail` bajt pri zapase `reserve`.
    // Vozvrashchaet 0, esli ne vlezaet ni odin. Chistaja arifmetika, nichego ne vydeljaet -
    // vyzyvaetsja do init(), chtoby raschjot mozhno bylo NAPECHATAT.
    static int auto_capacity(uint64_t avail, uint64_t reserve, std::size_t bytes_per_expert,
                             int n_layer, int spares, int n_expert);

    // Bajt na odnogo eksperta po trjom tenzoram sloja (gate + up + down).
    static std::size_t bytes_per_expert(const ggml_tensor* gate, const ggml_tensor* up,
                                        const ggml_tensor* down);

    // Odin raz: vydelit pamjat i zapomnit istochniki. Istochniki - tenzory MODELI (pod mmap
    // ili uzhe perepakovannye); iz nih i idjot kopirovanie v sloty.
    bool init(const ExpertStoreConfig& cfg,
              const std::vector<const ggml_tensor*>& gate,
              const std::vector<const ggml_tensor*>& up,
              const std::vector<const ggml_tensor*>& down,
              std::string* err);

    bool on() const { return cfg_.capacity > 0 && !layers_.empty(); }
    const ExpertStoreConfig& cfg() const { return cfg_; }
    const ExpertStoreStats&  stats() const { return st_; }
    std::size_t bytes() const { return bytes_; }
    std::size_t bytes_per_expert() const { return per_expert_; }
    bool repacked() const { return repacked_; }
    const std::string& repack_why() const { return repack_why_; }
    // Prjamoe chtenie fajla vkljucheno (mimo mmap) i skolko kuskov s diska prochitano.
    bool     direct() const { return direct_; }
    uint64_t direct_reads() const { return direct_reads_; }

    // Zatravka schjotchikov iz fajla kalibrovki. Format - sm. bench/expert_prior.py:
    // zagolovok int32[4] = {0x45585052, n_layer, n_expert, 1}, telo f32 [n_layer][n_expert].
    bool load_prior(const char* path, std::string* err);

    // Vybory dokumenta v schjotchiki. `ids` - poriadok [sloj][token][mesto], to est rovno
    // rsel harnessa; ili odin sloj odnogo tokena, esli n_tokens == 1.
    void observe_prefill(const int32_t* ids, int n_layer, int n_tokens, int n_used);

    // Pervichnaja zalivka: verhnie C po schjotchikam v sloty 0..C-1. Do zapuska chasov.
    bool prime(std::string* err);
    bool primed() const { return primed_; }

    // Uzel perevoda id -> slot dlja sloja il. Vstavljaetsja mezhdu top-k i mul_mat_id.
    ggml_tensor* slots_node(ggml_context* c, int il, ggml_tensor* sel, ggml_cgraph* gf);

    ggml_tensor* gate(int il) const { return layers_[std::size_t(il)].gate; }
    ggml_tensor* up(int il)   const { return layers_[std::size_t(il)].up; }
    ggml_tensor* down(int il) const { return layers_[std::size_t(il)].down; }

    // Konec tokena dekoda: schjot tokena i - kogda period nastupil - perestanovka nabora.
    void end_token();

    // Rezidentnyh na sloj (dolzhno byt ravno C posle pervoj perestanovki).
    int n_resident(int il) const;
    // Skolko razlichnyh ekspertov sloj voobshche videl (dlja "hvosta pervyh pojavlenij").
    int n_seen(int il) const;

    // Sverka odnogo slota s tenzorom modeli, bajt v bajt. Vozvrashchaet false i prichinu,
    // esli slot pust ili bajty rashodjatsja. Pri vkljuchennom repake sravnenie NEVOZMOZHNO -
    // ob etom i govorit prichina, a ne "soshlos".
    bool verify_slot(int il, int slot, std::string* why) const;

  private:
    struct Layer {
        ggml_tensor* gate = nullptr;
        ggml_tensor* up   = nullptr;
        ggml_tensor* down = nullptr;
        const ggml_tensor* src_gate = nullptr;
        const ggml_tensor* src_up   = nullptr;
        const ggml_tensor* src_down = nullptr;
        // Fajlovoe smeshchenie eksperta 0 po kazhdomu tenzoru (dlja prjamogo chtenija). Ekspert
        // id lezhit na off + id*nb[2], toch'-v-toch' kak src->data + id*nb2 pri mmap.
        uint64_t off_gate = 0;
        uint64_t off_up   = 0;
        uint64_t off_down = 0;
        ggml_backend_buffer_t buf = nullptr;
        std::vector<float>   score;     // [n_expert] zatravka + vybory
        std::vector<int32_t> slot_of;   // [n_expert] -> slot ili -1
        std::vector<int32_t> id_of;     // [S] -> id ili -1
        std::vector<uint8_t> want;      // [n_expert] rezidenten po poslednej perestanovke
        std::vector<uint64_t> touched;  // [S] nomer tokena, na kotorom slot zapolnjali
        std::vector<uint8_t> seen;      // [n_expert] vstrechalsja li v dokumente
        int  ring = 0;                  // kursor po zapasnym slotam
    };
    // Odna eta struktura na sloj: userdata u ggml_map_custom odin void*.
    struct Site { ExpertStore* self = nullptr; int il = 0; };

    static void map_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* ud);
    void do_map(int il, ggml_tensor* dst, const ggml_tensor* sel);
    // Kopirovanie odnogo eksperta v slot: tri kuska, potom - esli prosili - repak slota.
    void fill_slot(Layer& L, int slot, int id);
    void refresh();
    int  pick_victim(Layer& L);
    // Prjamoe chtenie fajla mimo strannichnogo kesha: kusok [foff, foff+len) v dst. Diapazon
    // vyravnivaetsja po sektoru (4096) pod FILE_FLAG_NO_BUFFERING, chitaetsja v io_buf_, nuzhnyj
    // poddiapazon kopiruetsja v dst. Vozvrashchaet false pri oshibke I/O.
    bool read_chunk(uint64_t foff, std::size_t len, void* dst);
    // Odin raz v init: otkryt gguf, najti fajlovye smeshchenija tenzorov ekspertov po imenam,
    // otkryt fajl s NO_BUFFERING, vydelit vyrovnennyj io_buf_. err - prichina otkaza.
    bool open_direct(std::string* err);

    ExpertStoreConfig  cfg_;
    ExpertStoreStats   st_;
    std::vector<Layer> layers_;
    std::vector<Site>  sites_;
    ggml_context*      ctx_   = nullptr;
    std::size_t        bytes_ = 0;
    std::size_t        per_expert_ = 0;
    int                n_slots_ = 0;
    uint64_t           token_ = 0;
    bool               repacked_ = false;
    std::string        repack_why_;
    bool               primed_ = false;
    // Idjot pervichnaja zalivka: sloty zapolnjajutsja bez poslotovogo repaka, potomu chto
    // prime() perepakuet celye tenzory odnim vyzovom v konce. Repak NE idempotenten.
    bool               bulk_ = false;
    // Prjamoe chtenie fajla vmesto memcpy iz mmap (shag 4, protiv dvojnoj pamjati).
    bool               direct_    = false;
    void*              fh_        = nullptr;   // HANDLE fajla, otkryt s NO_BUFFERING
    void*              io_buf_    = nullptr;   // vyrovnennyj po 4096 bufer pod odin kusok
    std::size_t        io_cap_    = 0;
    uint64_t           file_size_ = 0;
    uint64_t           direct_reads_ = 0;      // skolko kuskov prochitano s diska
};

}  // namespace memex
