# Threshold Analiz Raporu (11 Temmuz)

> [!CAUTION]
> **Historical, invalidated threshold.** `0.35` came from legacy row-level OOF
> predictions. Production uses the accepted full grouped threshold recorded in
> `docs/threshold_analysis.md`; the historical value below is not approved.
> Current runner: `python scripts/analysis/run_threshold_analysis.py`; new
> results are written to `docs/threshold_analysis.md`.

**Hazırlayan:** Ahmet Emin Işın  
**Tarih:** 11 Temmuz 2026  
**Yöntem:** 5-Fold OOF tahminleri üzerinde threshold taraması

---

## 1. Özet Sonuç

| Metrik | Varsayılan (0.5) | **Optimal** |
|---|---|---|
| Threshold | 0.5 | **0.35** |
| Macro-F1 | 0.9605 | **0.9613** |
| F1 (Pozitif) | 0.9405 | 0.9419 |
| F1 (Negatif) | 0.9805 | 0.9808 |
| Precision | 0.9537 | 0.9462 |
| Recall | 0.9277 | 0.9377 |

> [!NOTE]
> Threshold optimizasyonu ile **+0.0008** Macro-F1 kazanımı. Optimal threshold: **0.35**

---

## 2. Neden 0.5 Optimal Değil?

Eğitim setinde **3:1 negatif oran** kullanılıyor:
- Pozitif örnek: %25
- Negatif örnek: %75

Model bu dengesizliği öğrenerek tahminlerini aşağı kaydırır.
Bu yüzden optimal threshold 0.5'ten **düşük** çıkar.

---

## 3. Tam Threshold Tablosu

| Threshold | Macro-F1 | F1+ | F1- | Precision | Recall |
|---|---|---|---|---|---|
| 0.1 | 0.9425 | 0.9148 | 0.9702 | 0.8838 | 0.9480 |
| 0.15 | 0.9534 | 0.9305 | 0.9764 | 0.9174 | 0.9440 |
| 0.2 | 0.9576 | 0.9366 | 0.9787 | 0.9306 | 0.9427 |
| 0.25 | 0.9595 | 0.9393 | 0.9797 | 0.9367 | 0.9420 |
| 0.3 | 0.9601 | 0.9401 | 0.9801 | 0.9406 | 0.9397 |
| 0.35 | 0.9613 | 0.9419 | 0.9808 | 0.9462 | 0.9377 | << OPTIMAL
| 0.4 | 0.9610 | 0.9414 | 0.9807 | 0.9483 | 0.9347 |
| 0.45 | 0.9612 | 0.9415 | 0.9808 | 0.9520 | 0.9313 |
| 0.5 | 0.9605 | 0.9405 | 0.9805 | 0.9537 | 0.9277 | << Varsayilan
| 0.55 | 0.9602 | 0.9400 | 0.9804 | 0.9559 | 0.9247 |
| 0.6 | 0.9592 | 0.9384 | 0.9800 | 0.9593 | 0.9183 |
| 0.65 | 0.9579 | 0.9364 | 0.9795 | 0.9614 | 0.9127 |
| 0.7 | 0.9573 | 0.9353 | 0.9793 | 0.9646 | 0.9077 |
| 0.75 | 0.9548 | 0.9314 | 0.9782 | 0.9673 | 0.8980 |
| 0.8 | 0.9506 | 0.9248 | 0.9764 | 0.9703 | 0.8833 |
| 0.85 | 0.9434 | 0.9134 | 0.9734 | 0.9743 | 0.8597 |
| 0.9 | 0.9325 | 0.8960 | 0.9689 | 0.9791 | 0.8260 |

---

## 4. Sonraki Adımlar

- **Geçersiz eski öneri:** `0.35` üretimde kullanılmamalıdır.
- Tam eğitim ve BM25 dahil güncel sonuç için `docs/threshold_analysis.md` kullanılır.

*Ham CSV: `outputs/threshold_analizi.csv`*
