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

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
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

    // TRJOHUROVNEVOE HRANILISHCHE (ideja polzovatelja). Krome rezidentnyh slotov v OZU (uroven 1)
    // zavoditsja "tjoploe hranilishche" - kopija chasti NErezidentnyh ekspertov v otdelnom fajle
    // na BYSTROM diske (SSD, uroven 2). Ostalnye NErezidentnye ostajutsja v ishodnom gguf na
    // MEDLENNOM diske (HDD, uroven 3). Promah chitaetsja s togo urovnja, gde ekspert lezhit:
    // SSD ~2,7 ms, HDD ~23 ms. warm_dir pust => tjoplogo urovnja net (vsjo nerezidentnoe = HDD).
    std::string warm_dir;        // --warm-dir: katalog pod tjoplyj fajl na SSD
    double      warm_cap_gib = 0.0;  // --warm-cap: predel razmera tjoplogo fajla (GiB)
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
    // PREDZAGRUZKA PO R1 (shag 5). Vsjo schitaetsja na potoke grafa, tak chto delitel st_ ne
    // trogaetsja iz I/O-potoka.
    uint64_t pref_predicted = 0; // top-B idov, predskazannyh po R1 (summa po slojam za token)
    uint64_t pref_issued    = 0; // iz nih te, kotoryh ne bylo v slotah => postavleny v chtenie
    uint64_t pref_used       = 0; // popadanij, chej slot byl zapolnen imenno predzagruzkoj
    uint64_t pref_waits      = 0; // popadanij, gde chtenie ne uspelo i prishlos zhdat sinhronno
    double   ms_pref_wait    = 0.0; // vremja etih ozhidanij (async promah, ne spryatannyj)

    // TRJOHUROVNEVYJ UCHJOT (graf-potok). Sinhronnye promahi i bajty razbivajutsja po urovnju,
    // otkuda ekspert prochitan: SSD (tjoplyj fajl) ili HDD (ishodnyj gguf).
    uint64_t sync_ssd  = 0;   // sinhronnyh promahov, prochitannyh s tjoplogo urovnja (SSD)
    uint64_t sync_hdd  = 0;   // sinhronnyh promahov, prochitannyh s holodnogo urovnja (HDD)
    uint64_t bytes_ssd = 0;   // bajt, prochitannyh graf-potokom s SSD (zalivka+refresh+promah+sync-pref)
    uint64_t bytes_hdd = 0;   // bajt, prochitannyh graf-potokom s HDD
    uint64_t wait_ssd  = 0;   // async-ozhidanij, gde nedochitannyj slot byl s SSD
    uint64_t wait_hdd  = 0;   // async-ozhidanij, gde nedochitannyj slot byl s HDD

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

    // PREDZAGRUZKA PO R1 (shag 5).
    //
    // Popravki R1: logity sloja tgt ~ [router_tgt * x_l ; 1] * Wc_tgt, gde Wc_tgt eto
    // matrica [n_in=513][n_out=512]. Fajl r1_corr_kK.bin: zagolovok int32[4] =
    // {n_layer, 513, 512, K}, telo f16 [n_layer][513][512]; dlja tgt < K - tozhdestvo.
    // K berjotsja iz zagolovka. budget - skolko idov brat (top-B). sync - chitat sinhronno
    // pryamo v uzle (dlja A/B: NE prjachet chtenie, tak kak potok grafa stoit), inache -
    // otdat na potok vvoda-vyvoda i zhdat tolko v tochke ispolzovanija, esli ne uspelo.
    // Otsutstvie fajla ili nesovpadenie form - otkaz vsluh (false + err).
    // conf_hi / conf_lo - DVA POROGA UVERENNOSTI po rangu top-B (ideja polzovatelja): rang
    // predskazannogo eksperta < hi => predzagruzka objazatelna S LJUBOGO urovnja; hi <= rang <
    // lo => predzagruzka tolko esli ekspert na HDD (medlennyj uroven, gde uprezhdenie cenno) i
    // ochered ne zabita; rang >= lo => ne trogat. Znachenie <= 1.0 - dolja B, > 1.0 - absoljutnyj
    // rang. Po umolchaniju hi = lo = 1.0 (== B): staroe povedenie (ves top-B s ljubogo urovnja).
    // hdd_lead - na skolko sloev RANSHE nachinat predzagruzku HDD-ekspertov (23 ms nado prjatat
    // glubzhe): dobavljaetsja vtoroj uzel na sloj l, celjashchij v l+K+lead i vydajushchij tolko
    // HDD-eksperty (popravka Wc[l+K+lead] primenjaetsja k x_l - priblizhenie, K v fajle fiksirovan).
    bool load_prefetch(const char* path, int budget, bool sync, double conf_hi, double conf_lo,
                       int hdd_lead, std::string* err);
    bool prefetch_on() const { return pref_on_; }
    int  prefetch_k() const { return pref_k_; }
    int  prefetch_budget() const { return pref_budget_; }
    bool prefetch_sync() const { return pref_sync_; }
    int  prefetch_hdd_lead() const { return pref_hdd_lead_; }
    double prefetch_conf_hi() const { return pref_conf_hi_; }
    double prefetch_conf_lo() const { return pref_conf_lo_; }

    // TRJOHUROVNEVOE HRANILISHCHE (ideja polzovatelja).
    bool     warm_on() const { return warm_on_; }
    int      warm_per_layer() const { return warm_w_; }      // tjoplyh ekspertov na sloj
    uint64_t warm_bytes() const { return warm_size_; }        // razmer tjoplogo fajla
    const std::string& warm_path() const { return warm_path_; }
    double   warm_build_ms() const { return warm_build_ms_; }
    // Skolko iz nerezidentnyh ekspertov videli s SSD i s HDD (async-potok).
    uint64_t pref_reads_ssd() const { return pref_reads_ssd_.load(); }
    uint64_t pref_reads_hdd() const { return pref_reads_hdd_.load(); }
    uint64_t pref_bytes_ssd() const { return pref_bytes_ssd_.load(); }
    uint64_t pref_bytes_hdd() const { return pref_bytes_hdd_.load(); }

    // Uzel predskazanija+predzagruzki dlja celevogo sloja tgt. Vhod - syrye logity R0
    // (router_tgt * x_l, [n_expert]); uzel primenjaet popravku Wc_tgt, berjot top-B i stavit
    // nerezidentnye v chtenie v zapasnye sloty tgt. Vozvrata net (pobochnoe dejstvie), poetomu
    // uzel javno raskryvaetsja v graf. hdd_only - uzel HDD-uprezhdenija (hdd_lead): vydajot
    // TOLKO ekspertov na HDD-urovne, chtoby dat im bolshe fory.
    ggml_tensor* prefetch_node(ggml_context* c, ggml_cgraph* gf, int tgt,
                               ggml_tensor* pred_logits, bool hdd_only = false);

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
        // PREDZAGRUZKA: filling[s]=1 poka I/O-potok pishet bajty slota s (dostup pod mu_);
        // pref_tag[s] = nomer tokena, na kotorom slot zapolnjala imenno predzagruzka.
        std::vector<uint8_t>  filling;
        std::vector<uint64_t> pref_tag;
        // TRJOHUROVNEVOE: smeshchenie eksperta v TJOPLOM fajle (SSD) ili -1, esli on ne tjoplyj
        // (togda promah chitaetsja s HDD-gguf). [n_expert]. Rasklad fiksiruetsja v prime po
        // rangu chastoty: verhnie C - rezidentnye (OZU), sledujushchie W - tjoplye (SSD).
        std::vector<int64_t> warm_base;
    };
    // Odna eta struktura na sloj: userdata u ggml_map_custom odin void*.
    struct Site { ExpertStore* self = nullptr; int il = 0; };

    static void map_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth, void* ud);
    void do_map(int il, ggml_tensor* dst, const ggml_tensor* sel);
    // Kopirovanie odnogo eksperta v slot: tri kuska, potom - esli prosili - repak slota.
    void fill_slot(Layer& L, int slot, int id);
    void refresh();
    int  pick_victim(Layer& L);

    // --- PREDZAGRUZKA PO R1 (shag 5) ---
    static void pref_map_op(ggml_tensor* dst, const ggml_tensor* a, int ith, int nth,
                            void* ud);
    void do_prefetch(int tgt, const ggml_tensor* pred_logits, bool hdd_only);
    // Postavit eksperta id sloja tgt v zapasnoj slot: sinhronno (fill_slot) ili cherez potok.
    void prefetch_expert(Layer& L, int tgt, int id);
    // Chtenie treh kuskov eksperta id v slot cherez UKAZANNYE hendl+bufer (potok grafa i
    // I/O-potok imejut svoi, chtoby ne delit stateful fajlovyj ukazatel).
    void read_slot(Layer& L, int slot, int id, void* fh, void* buf, std::size_t cap,
                   std::atomic<uint64_t>* reads, std::atomic<uint64_t>* rbytes);
    void worker_main();
    void drain_prefetch();
    bool open_pref_handle(std::string* err);   // vtoroj hendl+bufer pod I/O-potok

    struct PrefSite { ExpertStore* self = nullptr; int tgt = 0; bool hdd_only = false; };
    struct Job { int il = 0; int slot = 0; int id = 0; };
    // Prjamoe chtenie fajla mimo strannichnogo kesha: kusok [foff, foff+len) v dst. Diapazon
    // vyravnivaetsja po sektoru (4096) pod FILE_FLAG_NO_BUFFERING, chitaetsja v io_buf_, nuzhnyj
    // poddiapazon kopiruetsja v dst. Vozvrashchaet false pri oshibke I/O.
    bool read_chunk(uint64_t foff, std::size_t len, void* dst);
    // Nizkourovnevoe sektorno-vyrovnennoe chtenie cherez ukazannye hendl i bufer (bez
    // schjotchikov): odin i tot zhe kod dlja potoka grafa (fh_/io_buf_) i I/O-potoka. fsize -
    // razmer imenno etogo fajla (gguf ili tjoplogo), nuzhen dlja kljona sektornogo hvosta.
    bool read_range(void* fh, void* buf, std::size_t cap, uint64_t fsize, uint64_t foff,
                    std::size_t len, void* dst);

    // --- TRJOHUROVNEVOE HRANILISHCHE ---
    // Sobrat tjoplyj fajl na SSD: warm_ids po slojam (verhnie posle rezidentnyh po chastote)
    // kopirujutsja iz gguf v otdelnyj fajl na warm_dir, ustanavlivaetsja L.warm_base. Potom
    // fajl otkryvaetsja na chtenie s NO_BUFFERING (dva hendla: graf i I/O-potok). err - prichina.
    bool build_warm(const std::vector<std::vector<int>>& warm_ids, std::string* err);
    // Uroven eksperta: true - tjoplyj (SSD), false - holodnyj (HDD).
    bool is_warm(const Layer& L, int id) const {
        return warm_on_ && id >= 0 && L.warm_base[std::size_t(id)] >= 0;
    }
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
    uint64_t           direct_reads_ = 0;      // skolko kuskov prochitano s diska (potok grafa)

    // --- TRJOHUROVNEVOE HRANILISHCHE (SSD-tjoplyj uroven) ---
    bool         warm_on_    = false;
    int          warm_w_     = 0;              // tjoplyh ekspertov na sloj
    std::string  warm_path_;                  // put tjoplogo fajla
    void*        fh_warm_      = nullptr;       // hendl tjoplogo fajla (potok grafa), NO_BUFFERING
    void*        fh_warm_pref_ = nullptr;       // hendl tjoplogo fajla (I/O-potok)
    uint64_t     warm_size_    = 0;             // razmer tjoplogo fajla (sektorno-vyrovnen)
    double       warm_build_ms_ = 0.0;         // vremja postrojki tjoplogo fajla
    // Vnutri odnogo tjoplogo eksperta gate/up/down lezhat podrjad; eti dliny nuzhny, chtoby
    // najti up i down po baze gate. Berutsja iz sloja 0 (u vseh sloev odinakovy).
    std::size_t  warm_nb_g_ = 0, warm_nb_u_ = 0, warm_nb_d_ = 0;

    // --- PREDZAGRUZKA PO R1 (shag 5) ---
    bool                    pref_on_     = false;
    bool                    pref_sync_   = false;   // chitat v uzle, ne na potoke (A/B)
    int                     pref_k_      = 0;       // K iz zagolovka fajla popravok
    int                     pref_budget_ = 16;      // B: skolko idov predzagruzhat
    double                  pref_conf_hi_ = 1.0;    // rang < hi (dolja B ili absolut) => ljuboj uroven
    double                  pref_conf_lo_ = 1.0;    // hi <= rang < lo => tolko HDD; >= lo => ne trogat
    int                     pref_hdd_lead_ = 0;     // na skolko sloev ranshe uprezhdat HDD
    int                     pref_n_in_   = 0;       // 513
    int                     pref_n_out_  = 0;       // 512 (== n_expert)
    std::vector<float>      pref_corr_;             // [n_layer * n_in * n_out] f32 (iz f16)
    std::vector<PrefSite>   pref_sites_;            // po celevomu sloju (osnovnoj, ljuboj uroven)
    std::vector<PrefSite>   pref_hdd_sites_;        // po celevomu sloju (HDD-uprezhdenie, hdd_lead)
    std::vector<float>      pref_score_;            // [n_expert], bufer scora (potok grafa)
    std::vector<int>        pref_idx_;              // [n_expert], bufer idov (potok grafa)
    // Vtoroj fajlovyj hendl i bufer - tolko dlja I/O-potoka.
    void*                   fh_pref_     = nullptr;
    void*                   io_buf_pref_ = nullptr;
    std::size_t             io_cap_pref_ = 0;
    std::atomic<uint64_t>   pref_reads_{0};         // kuskov prochitano I/O-potokom
    std::atomic<uint64_t>   pref_read_bytes_{0};
    // Razbivka chtenij I/O-potoka po urovnju (SSD-tjoplyj / HDD-gguf).
    std::atomic<uint64_t>   pref_reads_ssd_{0};
    std::atomic<uint64_t>   pref_reads_hdd_{0};
    std::atomic<uint64_t>   pref_bytes_ssd_{0};
    std::atomic<uint64_t>   pref_bytes_hdd_{0};
    // Ochered i sinhronizacija I/O-potoka.
    std::thread             worker_;
    std::mutex              mu_;
    std::condition_variable cv_;
    std::deque<Job>         jobs_;
    int                     inflight_    = 0;       // postavleno v chtenie, no ne zaversheno
    bool                    stop_        = false;
    bool                    worker_up_   = false;

  public:
    // Statistika predzagruzki dlja otchjota (I/O-potok).
    uint64_t pref_reads() const { return pref_reads_.load(); }
    uint64_t pref_read_bytes() const { return pref_read_bytes_.load(); }
};

}  // namespace memex
