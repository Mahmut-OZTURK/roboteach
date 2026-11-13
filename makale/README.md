# RoboTeach IEEE Araştırma Makalesi Paketi (TR & EN)

**Başlık (TR):** Fiziksel Temas Tabanlı Robotik Manipülasyonda Kapalı Döngü Nöro-Sembolik Deneyim Kalibrasyonu, Çok Bakışlı 3B Nokta Bulutu Algılama ve Kanonik Görev Ontolojisi  
**Title (EN):** Closed-Loop Neuro-Symbolic Experience Calibration, Multi-View 3D Point-Cloud Perception, and Canonical Task Ontology for Contact-Rich Robotic Manipulation  
**Yazarlar / Authors:** Muhammed Emin Çelik, Barış Kaya, Mahmut Öztürk, İsmail İşeri* (*Sorumlu Yazar / Corresponding Author)  
**Kurum / Affiliation:** Bilgisayar Mühendisliği Bölümü, Ondokuz Mayıs Üniversitesi, Samsun, Türkiye

## Paket İçeriği

| Dosya | Açıklama |
| :-- | :-- |
| `Kapali_Dongu_Noro_Sembolik_Robotik_Manipulasyon_IEEE_TR.docx` | İki sütunlu `Times New Roman` IEEE Transactions/Konferans formatında Türkçe Word makalesi |
| `Closed_Loop_Neuro_Symbolic_Robotic_Manipulation_IEEE_EN.docx` | İki sütunlu `Times New Roman` IEEE Transactions/Conference formatında İngilizce Word makalesi |
| `makale_tr.md` | Türkçe tam metin IEEE makale kaynağı (Denklem 1–8, Tablo I–V, 46 hakemli kaynak) |
| `makale_en.md` | İngilizce tam metin IEEE makale kaynağı (Equations 1–8, Tables I–V, 46 peer-reviewed references) |
| `latex/main.tex` | `IEEEtran` sınıfında Türkçe LaTeX makale kaynağı |
| `latex/main_en.tex` | `IEEEtran` sınıfında İngilizce LaTeX makale kaynağı |
| `latex/references.bib` | 46 hakemli dergi ve konferans yayını içeren BibTeX kaynakçası (sıfır ön-baskı) |
| `figures/*.svg` ve `figures/*.png` | Vektörel (`.svg`) ve yüksek çözünürlüklü (`2400 px` `.png`) TR/EN akademik şekiller (Görsel Özet + Şekil 1–3) |

## Deneysel Kıyaslamaları Yeniden Üretme

```bash
# 1) Birim ve entegrasyon testleri (17 test)
PYTHONPATH=.:venv/lib/python3.14/site-packages /usr/bin/pytest tests/ test_dryrun.py -q

# 2) 60 sahneli nominal ve gürültülü çok bakışlı 3B RGB-D algılama kıyaslaması
venv/bin/python -m benchmark.runner --mode perception --scenes 60

# 3) 6 görev ailesi x 4 katman canlı manipülasyon kıyaslaması (n=30 tohum/görev, 720 koşum)
venv/bin/python -m benchmark.runner --mode live --trials 30

# 4) Bireysel bileşen çıkarma ve alan rastgeleleştirme (DR) ablasyonu (1.080 koşum)
venv/bin/python -m benchmark.runner --mode component --trials 30

# 5) N=120 çapraz-aile (gpt-4.1) vs aynı-aile (gpt-6-luna) VLM hakem denetimi
venv/bin/python -m benchmark.runner --mode judge --episodes 20
```

## LaTeX Derleme Komutları

```bash
cd makale/latex
pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
pdflatex main_en.tex && bibtex main_en && pdflatex main_en.tex && pdflatex main_en.tex
```

