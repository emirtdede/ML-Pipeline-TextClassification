<div align="center">

# 🤖 ML Pipeline for Text Classification

[![](https://img.shields.io/badge/Language-English-blue?style=for-the-badge&logo=google-translate)](#english-version)
&nbsp;&nbsp;&nbsp;&nbsp;
[![](https://img.shields.io/badge/Dil-T%C3%BCrk%C3%A7e-red?style=for-the-badge&logo=google-translate)](#turkish-version)

---

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg?style=flat-square&logo=python)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Numpy](https://img.shields.io/badge/numpy-%23013243.svg?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](https://opensource.org/licenses/MIT)

</div>

---

<a id="english-version"></a>
# English Version

An end-to-end Machine Learning pipeline built in Python to train, tune, and evaluate multiple classification models for text categorization tasks. The system utilizes TF-IDF vectorization and Scikit-Learn's Grid Search cross-validation to search for optimal hyperparameter combinations, automatically saving the best-performing models.

## 🚀 Key Features

*   **🧹 Data Preprocessing & Cleaning**: Instantiates a structured `DataManager` to clean incoming text fields from JSON datasets.
*   **📐 TF-IDF Feature Vectorization**: Extracts numerical features from text using `TfidfVectorizer`, customizable with n-gram ranges and vocabulary limits.
*   **🛑 Stop-Words Management**: Leverages a custom `NLPManager` to filter out non-informative words (specifically tailored for Turkish datasets).
*   **🧠 Multiple ML Classifiers**: Support for training and comparison of 7 distinct machine learning algorithms:
    *   Decision Tree (`DTREE`)
    *   K-Nearest Neighbors (`KNN`)
    *   Stochastic Gradient Descent (`SGD`)
    *   Naive Bayes (`NB`)
    *   Support Vector Machines (`SVM`)
    *   Random Forest (`RANDFOREST`)
    *   Logistic Regression (`LOGREG`)
*   **⏱️ Hyperparameter Tuning (Grid Search)**: Automates grid search cross-validation to locate the best model configurations for feature extractions and classifier depths.
*   **💾 Model Serialization**: Automatically serializes and saves the best-tuned estimator pipelines as `.pkl` pickle files under the `models/` folder.

---

## 📁 Project Structure

```text
ML-Pipeline-TextClassification/
├── data/
│   └── news.json            # Dataset for classification
├── models/
│   └── best_dtree_model.pkl # Pickle file of the best tuned Decision Tree model
├── classification.py        # Enum definition for the classifiers
├── data.py                  # Core DataManager for cleaning and vector mapping
├── nlp.py                   # Natural Language Processing & custom stop words
├── train.py                 # Trainer wrapper facilitating Grid Search and serialization
├── vectorization.py         # Enum definition for the vectorizer
├── run.py                   # Main pipeline script configuring search grids and execution
└── README.md                # Project documentation
```

---

## ⚙️ Installation & Usage

### Prerequisites
*   Python 3.8+ installed.
*   Install required packages:
    ```bash
    pip install numpy scikit-learn
    ```

### Running the Pipeline
Executing the pipeline will run data preprocessing, initiate the configured Grid Searches (Decision Tree search is enabled by default), output execution metrics, and serialize the best estimator:
```bash
python run.py
```

---

## ⚖️ License
This project is licensed under the [MIT License](LICENSE).

---

<a id="turkish-version"></a>
# Türkçe Versiyon

Metin sınıflandırma görevleri için birden fazla makine öğrenmesi modelini eğitmek, ince ayar (hyperparameter tuning) yapmak ve değerlendirmek için Python'da oluşturulmuş uçtan uca bir Makine Öğrenmesi (ML) hattıdır. Sistem, TF-IDF vektörleştirmesini ve Scikit-Learn'ün Grid Search çapraz doğrulama altyapısını kullanarak en uygun hiperparametre kombinasyonlarını arar ve en iyi performans gösteren modelleri otomatik olarak kaydeder.

## 🚀 Öne Çıkan Özellikler

*   **🧹 Veri Ön İşleme ve Temizleme**: Girdi metin alanlarını JSON veri kümelerinden temizleyen ve yapılandıran bir `DataManager` yapısı.
*   **📐 TF-IDF Özellik Vektörleştirmesi**: Metinleri n-gram aralıkları ve kelime sınırı özellikleri sunan `TfidfVectorizer` ile sayısal özniteliklere dönüştürür.
*   **🛑 Stop-Words (Etkisiz Kelimeler) Yönetimi**: Metinlerdeki bilgi içermeyen kelimeleri filtreleyen (özellikle Türkçe veri kümelerine göre uyarlanmış) özelleştirilmiş bir `NLPManager` sınıfı.
*   **🧠 Çoklu ML Sınıflandırıcı Desteği**: 7 farklı makine öğrenmesi algoritmasının eğitilmesi ve karşılaştırılmasını destekler:
    *   Karar Ağacı (`DTREE`)
    *   K-En Yakın Komşu (`KNN`)
    *   Stokastik Gradyan İnişi (`SGD`)
    *   Naive Bayes (`NB`)
    *   Destek Vektör Makineleri (`SVM`)
    *   Rastgele Orman (`RANDFOREST`)
    *   Lojistik Regresyon (`LOGREG`)
*   **⏱️ Hiperparametre Ayarlama (Grid Search)**: Öznitelik çıkarma parametreleri ve sınıflandırıcı derinlikleri için en iyi model konfigürasyonlarını bulmak amacıyla ızgara araması (Grid Search) çapraz doğrulamasını otomatikleştirir.
*   **💾 Model Kaydetme**: En iyi performansa sahip eğitilmiş modelleri `models/` klasörü altına `.pkl` (pickle) formatında otomatik olarak kaydeder.

---

## 📁 Proje Yapısı

```text
ML-Pipeline-TextClassification/
├── data/
│   └── news.json            # Sınıflandırma için kullanılacak veri kümesi
├── models/
│   └── best_dtree_model.pkl # En iyi Karar Ağacı modelinin pickle dosyası
├── classification.py        # Sınıflandırıcı tiplerinin Enum tanımlaması
├── data.py                  # Veri temizleme ve vektör eşlemesini yürüten DataManager sınıfı
├── nlp.py                   # Türkçe etkisiz kelimeleri içeren nlp yöneticisi
├── train.py                 # Grid Search ve model kaydetme işlemlerini yürüten Trainer sınıfı
├── vectorization.py         # Vektörleştirici tiplerinin Enum tanımlaması
├── run.py                   # Arama ızgaralarını yapılandıran ve hattı başlatan ana betik
└── README.md                # Proje dökümantasyonu
```

---

## ⚙️ Kurulum ve Kullanım

### Gereksinimler
*   Python 3.8+ kurulmalıdır.
*   Gerekli paketleri yükleyin:
    ```bash
    pip install numpy scikit-learn
    ```

### Boru Hattını Çalıştırma
Ana betiği çalıştırmak; veri ön işlemeyi başlatır, yapılandırılmış Grid Search aramalarını (varsayılan olarak Karar Ağacı aktiftir) yürütür ve en iyi modeli pikle dosyası olarak kaydeder:
```bash
python run.py
```

---

## ⚖️ Lisans
Bu proje [MIT Lisansı](LICENSE) kapsamında lisanslanmıştır.
