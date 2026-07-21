# Error Taxonomy

Errors are generated with fold-specific model weights and thresholds selected without the evaluated fold. Taxonomy labels describe observable feature evidence; they do not assert root cause.

- Artifact mode: `full`
- Selected candidate: `weighted_blend`
- Cross-fitted Macro-F1: `0.837508`
- False positives: `68,740`
- False negatives: `71,467`

## Distribution

| Error type | Observed signal | Rows |
|---|---|---:|
| FN | COLOR_CONFLICT | 1,189 |
| FN | DEMOGRAPHIC_CONFLICT | 607 |
| FN | MATERIAL_CONFLICT | 251 |
| FN | MODEL_CODE_CONFLICT | 303 |
| FN | NO_LEXICAL_EVIDENCE | 9,105 |
| FN | OTHER | 60,012 |
| FP | COLOR_CONFLICT | 758 |
| FP | DEMOGRAPHIC_CONFLICT | 231 |
| FP | LEXICAL_DECOY | 45,185 |
| FP | MATERIAL_CONFLICT | 223 |
| FP | MODEL_CODE_CONFLICT | 81 |
| FP | NO_LEXICAL_EVIDENCE | 8,349 |
| FP | OTHER | 13,912 |
| FP | SIZE_CONFLICT | 1 |

## Highest-confidence False Positives

| Query | Product title | Probability | Signal |
|---|---|---:|---|
| u.s. polo assn. erkek çocuk | erkek çocuk koyu yeşil eşofman takımı 50318686-vr101 | 0.996936 | OTHER |
| koltuk örtüsü | jakar şönil kaymaz koltuk örtüsü şalı standart en 175cm x boy 300cm kollarıda ka | 0.996695 | LEXICAL_DECOY |
| greyder erkek ayakkabı | erkek koyu kahverengi hakiki deri chelsea bot 3k1ob16281 | 0.996039 | OTHER |
| çift kişilik yorgan | salkim yün yorgan % 100 doğal merinos yünü çift kişilik 195x215 | 0.995629 | LEXICAL_DECOY |
| zigon | kırlangıç zigon sehpa beyaz renk 4'lü set, beyaz ahşap ayak, iç içe geçebilir | 0.995514 | LEXICAL_DECOY |
| çatal kaşık bıçak | menashop silikon maşa 999662 | 0.995043 | OTHER |
| tek kişilik çarşaf takımı | bebek boy lastikli çarşaf seti - cartoon serisi - paw patrol | 0.994879 | OTHER |
| adidas erkek spor ayakkabı | erkek koşu ayakkabısı cloudfoam step js2912 | 0.994631 | OTHER |
| koltuk örtüsü | jakarli 3kişilik koltuk örtüsü , çekyat kanepe kılıfı. streç lastikli esnek (ürü | 0.994461 | LEXICAL_DECOY |
| dikiş makinesi aksesuarı | 33,5 no perçin (9mm) kalıp ve kapsülleri deri el aletleri | 0.994392 | OTHER |

## Lowest-confidence False Negatives

| Query | Product title | Probability | Signal |
|---|---|---:|---|
| içecek hazırlama | lattego 5500 serisi full otomatik en son model espresso capuccino (kapiçino) mak | 0.000051 | NO_LEXICAL_EVIDENCE |
| mini çanta | gül kurusu simli makyaj çantası - küçük boy | 0.000055 | NO_LEXICAL_EVIDENCE |
| modern salon duvar dekorasyonu | çerçeveli tablo seti, 3 parça çerçeveli tablo | 0.000075 | NO_LEXICAL_EVIDENCE |
| 1 yaş doğum günü | 80 adet krom gold beyaz deniz kumu balon seti | 0.000077 | NO_LEXICAL_EVIDENCE |
| metal kapaklı cam kavanoz | citronhaj, baharat kavanozu 4'lü set 350 ml | 0.000082 | NO_LEXICAL_EVIDENCE |
| elektrikli mangal ve barbekü | mangalet siyah granit bbq a333-02 | 0.000088 | OTHER |
| masa üstü dekoratif obje | aranjmanlı saksı as0110-35 | 0.000090 | NO_LEXICAL_EVIDENCE |
| hediyelik çikolata kutusu | kutu 25*15 *5 c.m hedıyelık cıkolata kutsu gold renk | 0.000102 | NO_LEXICAL_EVIDENCE |
| çocuk odası dekorasyonu | harita askılı kanvas poster | 0.000104 | NO_LEXICAL_EVIDENCE |
| kahverengi kalem | smokey göz kalemi (kahve) - smoky eyes waterproof eyeliner - 002 coolest brown - | 0.000118 | COLOR_CONFLICT |

The row-level evidence is stored in `outputs/error_taxonomy.csv`.
