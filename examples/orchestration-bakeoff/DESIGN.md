# Orchestration Bake-off — Tasarım Dokümanı

**Puras skill agent'ı** vs **LangGraph deterministik pipeline'ı**, aynı oyunda,
yüzlerce kez, yan yana.

> Durum: TASARIM (kod yok). Onaylanınca Faz 0'dan başlanır.
> Hedef branch: `claude/agentic-orchestration-demo-s94k29` (her iki repo).

---

## 1. Amaç ve tez

Puras'ın iddiası: **orkestrasyonu önceden elle çizmek (deterministik graf),
karşına çıkan öngörülemeyen sorunu çözmek için modele güvenmemek demektir.**
LangGraph/LangChain (ve Anthropic'in "managed" yaklaşımı) tam da bunu yapar —
akışı bir `StateGraph`/DAG olarak sabitler, modele mümkün olduğunca az karar
bırakır. Bu, **happy path'te** mükemmeldir ama grafın bir düğümü/dalı olarak
öngörülmemiş bir durum geldiğinde takılır.

Puras'ın `worker/agent_runner.py`'daki tek agent loop'u tersini yapar: modele
`bash` + dosya tool'ları + alt-agent'lar verip "hedefe nasıl ulaşacağına sen
karar ver" der. Bir sorunla karşılaşınca model farklı yollar deneyebilir,
durumu yeniden okuyabilir, edge case'i doğaçlama çözebilir.

**Bu demonun işi bu farkı kanıtlamak** — tek bir anekdotla değil, istatistikle.

### Tezin doğrulanabilir hali (hipotez)

> Bozulma (perturbation) oranı 0 iken iki taraf da ~%100 başarılı olur.
> Bozulma oranı arttıkça **deterministik pipeline'ın başarı oranı dik düşer**,
> **agent'ın başarı oranı yumuşak düşer**. İki eğri arasındaki alan = Puras'ın
> dayanıklılık (robustness) avantajı.

Eğer bu hipotez yanlış çıkarsa (agent da dik düşerse) bunu da dürüstçe
raporlarız — demo bir kanıt aracıdır, reklam panosu değil.

---

## 2. Neden oyun, neden bu oyun

Görsel, anlaşılır, objektif skorlanabilir ve "edge case"i mekanik olarak
üretebilen bir ortam lazım. Seçim: **Mastermind / Wordle + "yaramaz ev sahibi"**.

- **Taban oyun:** Gizli bir kod/kelime var. Oyuncu tahmin eder, ev sahibi her
  tahmine geri bildirim verir (Wordle: harf renkleri; Mastermind: kaç doğru
  pozisyon/renk). Oyuncu N hak içinde kodu bulmaya çalışır.
- **Neden iyi:** Kuralları herkes bilir, ekranda yan yana göstermek kolay,
  başarı **objektif** (kod bulundu mu / kaç hakta), ve optimal deterministik
  çözücü (Knuth'un Mastermind algoritması, Wordle entropi çözücüsü) iyi
  bilindiği için **LangGraph tarafını dürüstçe güçlü kurabiliriz** (strawman
  riski yok).

### "Yaramaz ev sahibi" = edge case enjektörü

Asıl numara burada. Ev sahibi bazen kuralları **önceden haber vermeden**
değiştirir. Örnekler (tam taksonomi §5'te):

- Geri bildirim formatı değişir (renkler yerine emoji, ya da JSON yerine düz
  metin).
- Kod uzunluğu oyun ortasında değişir.
- Geri bildirime gürültü karışır (ara sıra bir ipucu yanlış).
- Yeni bir kısıt belirir ("artık tahminlerin palindrom olmalı").
- Ev sahibi geçici olarak yalan söyler, sonra düzeltir.

Deterministik çözücünün grafında bu durumlara karşılık gelen düğüm yoktur →
ya çöker ya da kör tahmin eder. Agent ise geri bildirimi okur, "bir şey
değişti" sonucuna varır, stratejisini yeniden kurar — **tezin canlı
gösterimi.**

---

## 3. Adalet garantileri (EN KRİTİK BÖLÜM)

Bir demonun en büyük riski "sen tasarladın, bilerek LangGraph'ı aptal yazdın"
şüphesidir. Demo ancak aşağıdakileri sağlarsa ikna edicidir. Bunlar
pazarlık konusu değil:

1. **Tek bir paylaşılan ortam.** Oyun motoru ve perturbation enjektörü
   **iki taraftan da bağımsız** tek bir modüldür. İki oyuncu da ona aynı
   arayüzden (aynı `observe()` / `guess()` çağrıları) erişir. Hiçbir oyuncu
   perturbation'ı özel olarak göremez.
2. **Steelman LangGraph.** Deterministik taraf, taban oyununu **optimal**
   çözen gerçek bir çözücüdür (Mastermind için minimax/Knuth, Wordle için
   entropi). Amacı kötü görünmek değil; sadece "graf dışı" durumda doğal
   olarak takılması.
3. **Kör ve rastgele perturbation.** Hangi turda, hangi tür bozulmanın
   geleceği **seed'li RNG** ile belirlenir; iki oyuncu da aynı seed dizisiyle
   aynı bozulmalara maruz kalır. Bozulmalar Puras lehine elle seçilmez.
4. **Aynı LLM bütçesi/modeli.** İki taraf da aynı model ailesini kullanır
   (örn. ikisi de `claude/sonnet-4-6`). LangGraph'ın düğümleri de istediği
   yerde LLM çağırabilir — sadece akışı sabittir. Token/latency loglanır.
5. **N×{50,100,500,1000} koşu.** Tek playthrough yok. Her perturbation
   oranı için çok sayıda koşu; sonuç bir dağılım, ortalama ve güven aralığı.
6. **Reprodüksiyon.** Her koşu seed'i, perturbation günlüğü ve tam transkript
   diske yazılır; sonuç tekrar üretilebilir, kimse "uydurma" diyemez.
7. **Aynı görev tanımı.** İki tarafa da aynı hedef verilir: "kodu bul".
   Perturbation'lar hakkında **hiçbir taraf** önceden bilgilendirilmez —
   ne agent prompt'unda, ne LangGraph node'unda. Adalet için ikisi de
   "saf" başlar.

> Not: LangGraph tarafına "perturbation olabilir" diye bir retry/recovery
> node'u eklemek **mümkündür ve teşvik edilir** — ama bu da tezin diğer yüzü:
> her yeni edge case için yeni bir düğüm eklemek gerekir (graf şişer), oysa
> agent yeni bir kod yazmadan adapte olur. İki varyant koşarız: "naif
> LangGraph" ve "elden geldiğince savunmacı LangGraph". İkisini de gösteririz.

---

## 4. Mimari ve dosya yapısı

```
examples/orchestration-bakeoff/
  DESIGN.md                  # bu doküman
  README.md                  # nasıl koşulur
  engine/
    game.py                  # taban oyun motoru (Mastermind+Wordle), oyuncudan bağımsız
    perturbations.py         # edge-case taksonomisi + seed'li enjektör
    protocol.py              # Player arayüzü: observe()/guess() sözleşmesi
    scoring.py               # skor, recovery-rate, robustness eğrisi
  players/
    puras_player/            # PURAS TARAFI — bir skillpack
      codebreaker/
        SKILL.md
        skill.yaml
        tools/
          make_guess.py      # tahmini motora gönderir, geri bildirimi alır
          analyze.py         # (ops.) deterministik ipucu/eleme yardımcısı
        evals/
          cases.jsonl
      puras.yaml
    langgraph_player/        # DETERMİNİSTİK TARAF
      naive_solver.py        # Knuth/entropi çözücü, sabit StateGraph
      defensive_solver.py    # + recovery node'ları (graf şişmesini göstermek için)
      requirements.txt
  harness/
    run_bakeoff.py           # iki oyuncuyu N kez, P perturbation oranıyla koşar
    visualize.py             # yan-yana oynatım + robustness eğrisi grafiği
  results/                   # koşu çıktıları, seed'ler, transkriptler (gitignore)
```

### Akış

```
                 ┌─────────────────────────┐
                 │   engine/game.py        │  gizli kod, geri bildirim
                 │   + perturbations.py    │  (seed'li, oyuncudan bağımsız)
                 └───────────┬─────────────┘
            observe()/guess()│ (protocol.py — tek sözleşme)
            ┌────────────────┴────────────────┐
            ▼                                  ▼
   ┌──────────────────┐              ┌────────────────────┐
   │ Puras skill      │              │ LangGraph StateGraph│
   │ (agent loop)     │              │ (sabit akış)        │
   └──────────────────┘              └────────────────────┘
            │                                  │
            └──────────────┬───────────────────┘
                           ▼
                   harness/scoring.py
              (win-rate vs perturbation eğrisi)
```

### Protocol sözleşmesi (`protocol.py`)

İki oyuncuyu kıyaslanabilir tutan tek arayüz. Oyuncu sadece şunu görür:

```python
class Player(Protocol):
    def reset(self, rules: GameView) -> None: ...
    def next_guess(self, history: list[Turn]) -> str: ...

# GameView ve Turn, perturbation'ı GİZLEMEZ ama ETİKETLEMEZ de:
# format değiştiyse oyuncu bunu geri bildirimden kendi anlamak zorunda.
```

Kritik: motor, "bu turda perturbation var" diye bir bayrak **vermez**. İki
taraf da değişimi yalnızca gözlemden çıkarsamak zorundadır. Determinitik
çözücü bunu yapısı gereği yapamaz; agent yapabilir.

---

## 5. Perturbation taksonomisi (edge-case türleri)

Her biri seed'li RNG ile, belirli bir olasılıkla, belirli turlarda tetiklenir.
Zorluk arttıkça çeşitlilik ve sıklık artar.

| # | Perturbation | Ne yapar | Deterministik çözücüye etkisi |
|---|---|---|---|
| P1 | **Format kayması** | Geri bildirim renk→emoji→JSON→düz metin döner | Parser kırılır; agent yeni formatı okur |
| P2 | **Uzunluk değişimi** | Kod uzunluğu tur ortasında 5→6 olur | Sabit-uzunluk varsayımı çöker |
| P3 | **Gürültülü ipucu** | %X olasılıkla bir geri bildirim yanlış | Çözücü eler-eler boş kümeye düşer |
| P4 | **Yeni kısıt** | "Tahminler artık palindrom olmalı" | Graf'ta böyle bir dal yok |
| P5 | **Geçici yalan** | Ev sahibi k tur yalan söyler sonra düzeltir | Çözücü tutarsızlıkta kilitlenir |
| P6 | **Alfabe genişlemesi** | Geçerli semboller kümesi büyür | Arama uzayı varsayımı yanlışlanır |
| P7 | **Sessiz kural değişimi** | Skorlama "pozisyon"dan "varlık"a kayar | Çözücü yanlış modelle ilerler |

Her koşu için "perturbation oranı" = bir turda herhangi bir P'nin tetiklenme
olasılığı (0.0, 0.1, 0.25, 0.5). Eğri bu eksende çizilir.

---

## 6. Skorlama ve metrikler

Tek sayı değil, bir profil:

- **Win-rate** — kod N hak içinde bulundu mu? (birincil)
- **Robustness eğrisi** — win-rate vs perturbation oranı (birincil görsel).
- **Recovery-rate** — perturbation sonrası ilk turda yanlış gidip, sonraki
  k turda toparlama oranı. (Agent'ın "edge case çözme" yeteneğini en net
  yakalayan metrik.)
- **Verimlilik** — kazanırken kullanılan ortalama tahmin sayısı (perturbation
  yokken determinitik çözücü burada kazanmalı — dürüstlük göstergesi).
- **Maliyet/latency** — token ve süre. Agent muhtemelen daha pahalı; bunu
  saklamayız, "dayanıklılığın bedeli" olarak gösteririz.

Çıktı: her perturbation oranı için ortalama ± güven aralığı, bir CSV ve bir
grafik. Puras eval altyapısı (`puras eval --local`) skill tarafının kendi
sağlığını ayrıca ölçer.

---

## 7. Puras tarafı tasarımı (`players/puras_player/codebreaker`)

Mevcut skill yapısına birebir oturur (`greeter`/`content-repurposer` örnekleri
referans):

- **`skill.yaml`** — `input_schema`: oyun oturum kimliği + kurallar; `text_model:
  claude/sonnet-4-6`; `tools`: `make_guess` (tahmini motora yollar, ham geri
  bildirimi döner — yorumlamaz), opsiyonel `analyze`; `output_schema`: bulunan
  kod + kaç hak; `evals`: kod-bulundu check'i.
- **`SKILL.md`** — kasıtlı olarak **perturbation'lardan bahsetmez**. Sadece:
  "Bir kod kırma oyunu oynuyorsun. Her tur `make_guess` çağır, dönen geri
  bildirimi **dikkatle oku**, bir şey beklediğinden farklıysa varsayımlarını
  sorgula ve stratejini ona göre güncelle. Kodu bul." Agent'a *düşünme alanı*
  bırakmak, deterministik tarafa olan farkın kaynağıdır.
- **`tools/make_guess.py`** — saf, ince bir köprü: tahmini engine'e iletir,
  engine'in döndürdüğü (perturbe edilmiş olabilen) ham gözlemi geri verir.
  Tool **yorum yapmaz**; akıl agent'ta kalır.

---

## 8. LangGraph tarafı tasarımı (`players/langgraph_player`)

- **`naive_solver.py`** — `StateGraph`: `parse_feedback → update_candidates →
  pick_optimal_guess → submit → (loop | win)`. Mastermind için Knuth minimax,
  Wordle için entropi. Perturbation yokken **optimal**. Format/kural değişince
  `parse_feedback` veya `update_candidates` düğümü ya exception atar ya da
  sessizce yanlış model üretir → demonstrasyonun tam da göstermek istediği
  kırılma.
- **`defensive_solver.py`** — aynı grafa try/except, retry, "format yeniden
  algıla" düğümleri eklenir. Bazı perturbation'ları yakalar; ama her yeni P
  için yeni düğüm gerekir. Bu varyant tezin nüansını gösterir: determinizmle
  dayanıklılık, **graf karmaşıklığı** pahasına ve **yalnızca öngörülen** edge
  case'ler için kazanılır.

İki varyantı da koşup üçlü eğri çizeriz: agent / naif-LG / savunmacı-LG.

---

## 9. Çıktı ve görselleştirme (`harness/visualize.py`)

- **Yan-yana canlı oynatım** — sol: LangGraph, sağ: Puras. Her turda tahmin,
  geri bildirim, ve (perturbation tetiklenince) bir "⚡ kural değişti" işareti.
  İzleyici determinitik tarafın takıldığı, agent'ın toparladığı anı görür.
- **Robustness eğrisi** — N koşu sonrası win-rate vs perturbation oranı,
  üç çizgi. Demonun "kanıt" karesi budur.
- Terminal (rich) + opsiyonel basit web görünümü. Web görünümü gerekirse
  `purasbackend/frontend`'e küçük bir demo sayfası olarak bağlanabilir.

---

## 10. Riskler ve açık sorular

- **Agent da dik düşerse?** Hipotez yanlışlanabilir. O zaman ya oyun/SKILL.md
  agent'a yeterli alan bırakmıyordur, ya da tez bu görev için zayıftır. Sonucu
  dürüstçe raporlarız; gerekirse perturbation taksonomisini gözden geçiririz.
- **Maliyet.** 1000 koşu × LLM = gerçek para. Önce küçük modelle (haiku) ve
  düşük N ile kalibrasyon, sonra ölçek. `make_guess` mümkün olduğunca ucuz.
- **Adalet algısı.** §3'teki garantiler kodda görünür ve test edilebilir
  olmalı; "engine oyuncudan bağımsız" iddiasını bir teste bağlarız.
- **Görev seçimi tartışmaya açık.** Mastermind/Wordle net ama "yapay" bulunabilir.
  İleride aynı harness'la "bozuk tool'lu gerçek görev" ikinci bir senaryo olarak
  eklenebilir (taksonomi ve skorlama yeniden kullanılır).

## 11. Açık karar noktaları (uygulamadan önce)

1. **Oyun:** Mastermind mi, Wordle mı, ikisi de mi? (İkisi de aynı engine'e
   sığar; biriyle başlamak daha hızlı.)
2. **Repo yerleşimi:** `puras/examples/` (önerilen — runner ve örneklerin evi)
   mi, yoksa `purasbackend`'de bir demo dizini mi?
3. **Görselleştirme:** sadece terminal mi, yoksa web sayfası da mı?
4. **Ölçek:** kalibrasyon için başlangıç N ve model (haiku ile başlayıp
   sonra sonnet'e mi geçelim?).

## 12. Uygulama planı (fazlar)

- **Faz 0 — Engine + protocol.** `game.py`, `protocol.py`, perturbation iskeleti
  (henüz P1, P2). Oyuncusuz, testlerle.
- **Faz 1 — İki naif oyuncu.** Naif LangGraph çözücü + Puras `codebreaker`
  skill'i. Perturbation = 0'da ikisi de kazanmalı (doğrulama).
- **Faz 2 — Perturbation tam taksonomi + harness.** §5'in tamamı, N-koşu
  runner, scoring, ilk robustness eğrisi.
- **Faz 3 — Savunmacı LangGraph + görselleştirme.** Üçlü eğri, yan-yana
  oynatım.
- **Faz 4 — Cilalama.** README, reprodüksiyon, (ops.) web demo.
