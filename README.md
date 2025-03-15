MIT License

Copyright (c) 2025 Baha Büyükateş

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

Bu veri setindeki amaç, genel derin öğrenme (Deep Learning) yöntemlerini uygulamanın yanı sıra, Optuna gibi hiperparametre optimizasyonu (HPO) tekniklerinin önemini vurgulamaktır.

Bu kapsamda, modelin performansını artırmak için Optuna kullanılarak hiperparametre optimizasyonu yapılacak ve farklı hiperparametre kombinasyonlarının model üzerindeki etkisi incelenecektir. Hiperparametre optimizasyonunun model başarımı üzerindeki kritik rolü göz önünde bulundurularak, hassasiyet (accuracy) yerine F1-score metriğine odaklanılacaktır.

Böylece, yalnızca standart derin öğrenme modellerinin kurulumu değil, aynı zamanda hiperparametre optimizasyonu sürecinin nasıl yönetileceği de ele alınacaktır.



## **CNN (Convolutional Neural Network);**

Bir fotoğrafın CNN tarafından işlenmesi, **konvolüsyon, aktivasyon fonksiyonu, havuzlama (pooling), tam bağlantılı (fully connected) katmanlar ve sınıflandırma** gibi adımlardan geçer.

## **CNN’in Temel Katmanları ve İşleyiş Süreci**

### **1. Giriş (Input) Katmanı**

CNN, genellikle bir **RGB görüntü** (örneğin 150x150x3 boyutunda) alır.  
Buradaki **3** değeri, **kırmızı, yeşil ve mavi (RGB) kanallarını** temsil eder.  
Bu görüntü, **matris formatında** CNN modeline verilir.

### **2. Konvolüsyon (Convolution) Katmanı**

**Amaç:** Görüntüdeki temel özellikleri çıkarmak (kenarlar, dokular, şekiller).

- **Filtre (Kernel):** Küçük bir matris (örneğin 3x3 veya 5x5) kullanılarak görüntü taranır.
- Filtre, görüntü üzerinde **kaydırılarak** (stride) bölgesel özellikleri çıkarır.
- **Çıktı:** "Feature Map" (Özellik Haritası) adı verilen yeni bir matris üretilir.
- Birden fazla filtre kullanılarak farklı özellikler çıkarılabilir.

**Örnek:**  
Bir 3×3 kenar tespit filtresi kullanıldığında, görüntüdeki kenarlar vurgulanır.


### **3. Aktivasyon Fonksiyonu (ReLU - Rectified Linear Unit)**

 **Amaç:** Negatif değerleri kaldırarak modele doğrusal olmayan bir özellik kazandırmak.

- Konvolüsyon katmanının çıkardığı negatif değerleri **0’a** eşitler.
- ReLU kullanımı, modelin **daha iyi genelleştirme yapmasını** sağlar.

**Örnek:**  
Eğer bir piksel değeri **-10** ise, ReLU fonksiyonundan sonra **0** olur.  
Pozitif değerler değişmeden kalır.

Formül:

$ReLU(x)=max⁡(0,x)ReLU(x) = \max(0, x)ReLU(x)=max(0,x)$

### **4. Havuzlama (Pooling) Katmanı - Maksimum Havuzlama (Max Pooling)**

 **Amaç:** Görüntünün **boyutunu küçültmek** ve önemli bilgileri korumak.

**Max Pooling İşleyişi:**

- Belirli bir pencere (örneğin 2x2) görüntü üzerinde kaydırılır.
- Her bölgedeki **en büyük** değeri seçerek yeni bir özellik haritası oluşturur.
- Boyutu küçültür, işlem süresini hızlandırır ve overfitting’i azaltır.

**Örnek:**  
Aşağıdaki gibi bir 2×2 bölge düşünelim:

$\begin{bmatrix} 3 & 1 \\ 7 & 5 \end{bmatrix}$

Buradaki **en büyük değer 7** olduğu için, max pooling sonrası bu bölge **7** olur.

Alternatif olarak **Average Pooling** kullanılabilir, ancak Max Pooling genellikle daha iyi çalışır.

### **5. Daha Fazla Konvolüsyon ve Havuzlama Katmanları**

 **Amaç:** Derinleştikçe daha karmaşık özellikleri öğrenmek.

- İlk konvolüsyon katmanları **kenarları** tespit ederken,
- Daha derin katmanlar **şekilleri, nesneleri ve dokuları** öğrenir.

Örneğin:

- İlk katmanlar → **Düşük seviyeli özellikler (kenarlar, dokular)**
- Orta katmanlar → **Orta seviyeli özellikler (şekiller, nesneler)**
- Son katmanlar → **Yüksek seviyeli özellikler (göz, burun, araba vs.)**
### **6. Düzleştirme (Flatten) Katmanı**

 **Amaç:** 2D veriyi **1D vektör haline getirmek**.

- Havuzlama sonrası kalan özellik haritası **vektöre dönüştürülerek** tam bağlantılı katmanlara (Dense Layer) verilir.
- Örneğin, **7x7x512** boyutundaki bir çıktı, **1D vektöre** dönüştürülür.

### **7. Tam Bağlantılı (Fully Connected - Dense) Katmanlar**

 **Amaç:** Öğrenilen özellikleri kullanarak sınıflandırma yapmak.

- **Giriş:** Flatten edilmiş vektör.
- **İşleyiş:** Yoğun (dense) bağlantılı yapay sinir ağı katmanları kullanılır.
- **Aktivasyon Fonksiyonları:**
    - Ara katmanlarda **ReLU** kullanılır.
    - Son katmanda **Softmax** veya **Sigmoid** kullanılır.

**Örnek:**

- **Binary classification (İkili sınıflandırma)** için **sigmoid** kullanılır.
- **Çok sınıflı classification (Multi-class)** için **softmax** kullanılır.
### **8. Çıktı (Output) Katmanı**

 **Amaç:** Görüntünün hangi sınıfa ait olduğunu belirlemek.

- Son katmanda **Softmax** kullanılarak, her sınıf için olasılıklar hesaplanır.
- En yüksek olasılığa sahip sınıf seçilir.
## **CNN’in Katmanlarına Genel Bakış**

|Katman|İşlevi|
|---|---|
|**Giriş Katmanı**|Görüntüyü modele verir|
|**Konvolüsyon Katmanı**|Özellikleri çıkarır|
|**ReLU Aktivasyonu**|Negatif değerleri kaldırır|
|**Max Pooling**|Boyutu küçültür, önemli bilgileri korur|
|**Flatten**|2D'yi 1D’ye çevirir|
|**Tam Bağlantılı Katmanlar (Dense)**|Öğrenilen özellikleri kullanarak tahmin yapar|
|**Çıktı Katmanı (Softmax/Sigmoid)**|Sınıflandırma yapar|

---

## **Özetle CNN Nasıl Çalışır?**

1️⃣ **Giriş:** Görüntü modele verilir.  
2️⃣ **Konvolüsyon:** Görüntüdeki önemli özellikler çıkarılır.  
3️⃣ **Aktivasyon (ReLU):** Negatif değerler kaldırılır.  
4️⃣ **Havuzlama (Max Pooling):** Boyut küçültülür.  
5️⃣ **Tekrar Konvolüsyon + Pooling:** Daha soyut özellikler öğrenilir.  
6️⃣ **Düzleştirme (Flatten):** Matris, vektöre dönüştürülür.  
7️⃣ **Dense Katmanları:** Öğrenilen bilgiler sınıflandırılır.  
8️⃣ **Çıktı (Softmax):** En uygun sınıf tahmin edilir.

Bu süreç, CNN'nin bir fotoğrafı **tanımlamasını, analiz etmesini ve sınıflandırmasını** sağlar.

### **1. Giriş**

Beyin tümörlerinin erken teşhisi, hastalığın ilerlemesini önlemek ve tedavi sürecini optimize etmek için hayati önem taşır. Derin öğrenme modelleri, görüntü tabanlı tıbbi teşhislerde son derece başarılı sonuçlar vermektedir. Bu raporda, EfficientNetB0, Xception ve ResNet modellerinin beyin tümörü veriseti üzerinde neden kullanılabileceği detaylı olarak incelenecektir.

---

### **2. Modellerin Genel Tanıtımı**

#### **2.1. EfficientNetB0**

EfficientNet ailesi, model boyutunu optimize ederken doğruluğu yüksek tutmayı amaçlayan bir CNN (Convolutional Neural Network) mimarisidir.

- Model boyutu ve hesaplama ihtiyacı optimize edilmiştir.
    
- Transfer öğrenme için uygundur.
    
- Daha az parametre ile daha yüksek doğruluk elde edebilir.
    

#### **2.2. Xception**

Xception modeli, derinlik ayrışımına dayalı konvolüsyonları (depthwise separable convolutions) kullanarak hesaplama yükünü azaltan bir CNN mimarisidir.

- Daha etkin hesaplama için standart konvolüsyonlardan ayrışır.
    
- Parametre sayısı düşük olmasına rağmen doğruluk oranı yüksektir.
    
- Görüntü tabanlı tıbbi analizlerde başarılı sonuçlar vermektedir.
    

#### **2.3. ResNet**

ResNet (Residual Networks), derin sinir ağı (DNN) modellerinin öğrenme sürecindeki kayıplarını azaltmak için "skip connections" mekanizmasını kullanır.

- Daha derin ağlar oluşturmaya imkan tanır.
    
- "Vanishing Gradient" sorununu çözer.
    
- Tıbbi görüntüleme alanında yaygın olarak kullanılmaktadır.
    

---

### **3. Beyin Tümörü Teşhisinde Modellerin Kullanılması**

#### **3.1. EfficientNetB0**

EfficientNetB0 modeli, görüntüleme çalışmalarında yüksek performans göstermekte olup, beyin tümörü veriseti üzerinde de etkin bir şekilde kullanılabilir. Verisetindeki MRI görüntülerinin karmaşık yapısını anlamlandırmak için idealdir.

#### **3.2. Xception**

Beyin tümörleri genellikle ince detaylarla fark edilebilir. Xception modeli, derinlik ayrışımına dayalı konvolüsyon yapısıyla ince detayları yakalamada etkili olabilir. Özellikle glioblastom gibi detayları belirgin olmayan tümörlerde Xception'ın ayrışım yeteneği avantaj sağlar.

#### **3.3. ResNet**

ResNet, tıbbi görüntülemede yaygın olarak kullanılan modellerden biridir. Beyin tümörü tespiti için kullanıldığında, modelin derin mimarisi sayesinde karmaşık MRI görüntülerinden anlamlı özellikler çıkarabil

## **Optuna ve ImageNet:**

**ImageNet ağırlıkları nedir?**

ImageNet ağırlıkları, ImageNet veri seti üzerinde önceden eğitilmiş modellerin öğrendiği parametrelerdir.
Bu ağırlıklar, modelin genel özellikleri (kenar, doku, şekil gibi) tanımasına yardımcı olur; transfer learning uygulamalarında, yeni bir veri setine uyarlamadan önce modelin genel bilgilerini kullanmanıza olanak tanır.
Örneğin, Xception, EfficientNet, ResNet gibi popüler modeller, ImageNet üzerinde eğitilmiş ağırlıklarla sağlanır; böylece, küçük veri setlerinde bile iyi performans gösterebilirler.


Optuna’nın önerdiği hiperparametreler hakkında:

Kod örneklerimizde Optuna, modelin performansını optimize etmek için şu hiperparametreleri öneriyor:

learning_rate (öğrenme oranı)
dense_units (tam bağlı katmandaki nöron sayısı)
dropout_rate (dropout oranı)
batch_size (eğitim sırasında kullanılan batch boyutu)
Ek olarak, daha fazla hiperparametre de optimize edilebilir.
Örneğin:

Epoch sayısı: 
Eğitim döngülerinin sayısı (modelin öğrenme sürecini etkileyen önemli bir parametre olsa da, genellikle hiperparametre araması sırasında sabit bırakılır).

Optimizer türü: 
Adam, SGD, RMSprop gibi farklı optimizasyon algoritmaları seçilebilir.

Momentum, beta değerleri: 
Özellikle SGD veya Adam gibi optimizasyon algoritmalarında ek hiperparametreler bulunur.

Learning rate decay: 
Öğrenme oranının zamanla düşürülmesi gibi stratejiler.

Model mimarisi ile ilgili bazı parametreler: 
Örneğin, ek katman sayısı, **konvolüsyone**l filtre boyutları, kernel boyutları vb.

Veri artırma (augmentation) parametreleri: 
Eğitim sırasında kullanılacak veri artırma stratejileri ve oranları. Ancak, genellikle ilk aşamada temel hiperparametreleri optimize etmek daha pratik ve verimli olur. Daha sonraki aşamalarda model mimarisinde veya eğitim sürecinde ince ayarlar yapılarak diğer parametreler de optimize edilebilir.

Özetle, ImageNet ağırlıkları, modelin genel görüntü özelliklerini öğrenmiş halidir ve transfer learning için kullanılır. Optuna tarafından optimize edilen hiperparametreler temel ayarlardır; ancak model performansını daha da artırmak için daha fazla hiperparametre de optimize edilebilir.


EfficientNetB0 → Hızlı ve hafif modeller için.
EfficientNetB4 veya B5 → Dengeli performans için.
EfficientNetB7 → En yüksek doğruluk için, ancak çok daha fazla hesaplama gücü gerektirir.

**EfficientNet Relu değilde Swish:**
Swish, Google tarafından önerilen ve **ReLU’ya alternatif olarak geliştirilen** bir aktivasyon fonksiyonudur. **Matematiksel olarak şu şekilde tanımlanır:**

$Swish(x)=x⋅sigmoid(x)=x⋅11+e−xSwish(x) = x \cdot sigmoid(x) = x \cdot \frac{1}{1+e^{-x}}Swish(x)=x⋅sigmoid(x)=x⋅1+e−x1​$

Bu fonksiyon, giriş değerini **sigmoid ile çarparak** yumuşak bir geçiş sağlar. **Negatif giriş değerlerini tamamen sıfıra indirmez**, ancak küçük değerlere çekerek belli bir ölçüde korur.

**EfficientNet** → ReLU yerine Swish kullanarak daha iyi performans elde etmiştir.

### **Swish ve ReLU Arasındaki Farklar**

| **Özellik**                    | **ReLU (Rectified Linear Unit)**                                 | **Swish**                                                                 |
| ------------------------------ | ---------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **Matematiksel Tanım**         | $ReLU(x)=max(0,x)ReLU(x) = max(0, x)ReLU(x)=max(0,x)$            | $Swish(x)=x⋅sigmoid(x)Swish(x) = x \cdot sigmoid(x)Swish(x)=x⋅sigmoid(x)$ |
| **Negatif Değerler**           | 0 olarak kırpar                                                  | Küçük negatif değerleri korur                                             |
| **Sürekli türevlenebilir mi?** | Hayır, x=0x = 0x=0 noktasında süreksizlik vardır                 | Evet, türevlenebilir ve daha düzgün bir eğriye sahiptir                   |
| **Öğrenme Performansı**        | Genellikle hızlı öğrenir, ancak bazen "ölü nöron" problemi yaşar | Daha iyi doğruluk sağlar, ancak biraz daha fazla hesaplama gerektirir     |
| **Donanım Uyumu**              | Çok hızlı ve donanım için optimize edilmiştir                    | ReLU kadar hızlı değil, ancak modern GPU’lar için optimize edilebilir     |
| **Verimlilik**                 | Küçük modellerde hızlı çalışır                                   | Büyük modellerde daha iyi genel performans sağlar                         |
**Ölü Nöron Problemini Azaltır:** ReLU’da negatif değerler sıfıra sabitlendiği için bazı nöronlar "ölü" hale gelir.
**Sürekli ve Yumuşak Geçişler Sunar:** Swish fonksiyonu **daha esnek ve akışkan bir aktivasyon fonksiyonudur.**
**Her Modelde En İyi Sonuç Vermeyebilir:** Küçük ve hızlı modellerde **ReLU hâlâ daha iyi bir seçim olabilir.**



# **ResNet;**
**Derin Sinir Ağları’nda (DNN)** eğitim sırasında karşılaşılan **gradyan kaybolması (vanishing gradient)** problemine çözüm getiren bir konvolüsyonel sinir ağı (CNN) mimarisidir.

ResNet’in en büyük farkı, **residual bağlantılar (skip connections)** kullanarak derin ağların eğitimini kolaylaştırmasıdır.  
**Matematiksel Gösterim:**

Eğer standart bir CNN’de öğrenilmesi gereken çıktı **H(x)** ise, ResNet şu şekilde bir yapı kullanır:

$H(x)=F(x)+xH(x) = F(x) + xH(x)=F(x)+x$

Burada:

- **F(x)** → Öğrenilmesi gereken dönüşüm (Convolution, ReLU, vb.).
- **x** → Girdi verisinin **doğrudan aktarılan kısmı (skip connection)**.

Yani ağ, **tam sonucu öğrenmek yerine sadece "kalıntıyı" (residual) öğrenmeye çalışır**, bu da derin ağların eğitilmesini kolaylaştırır.

ResNet-50 ve daha derin modellerde **Bottleneck Block** (daha az parametre kullanarak verim artıran bir yapı) kullanılır.

- İlk **1x1 konvolüsyon** → Boyutu küçültür.
- **3x3 konvolüsyon** → Özellikleri öğrenir.
- Son **1x1 konvolüsyon** → Kanal sayısını artırır.
- **BatchNormalization** ve **skip connection** performansı artırır.

- **İlk konvolüsyon katmanı** → Büyük bir filtre (7x7) ile temel özellikleri çıkarır.
- **Pooling katmanı** → Boyutu küçülterek işlemleri hızlandırır.
- **Bottleneck bloklar** → Derin katmanların verimli çalışmasını sağlar.
- **GlobalAveragePooling2D** → Fully connected katmandan önce boyutu küçültür.
- **Softmax aktivasyonu** → Sonuçları sınıflandırmak için kullanılır.

**Daha Derin Ağlar Eğitebiliriz:** 100+ katmanlı ağlar bile stabilize edilebilir.  
**Gradyan Kaybolma Problemini Çözer:** Skip bağlantılar sayesinde derin ağlarda eğitim mümkün olur. 
 **Daha Yüksek Başarı Oranı Sunar:** ImageNet gibi veri setlerinde büyük başarı elde etmiştir.

$**Kod;**$
`import tensorflow as tf`
`from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input`
`from tensorflow.keras.models import Model`

`def resnet_basic_block(x, filters):`
    `shortcut = x  # Skip Connection için giriş verisini saklıyoruz.`

    `# 1. Konvolüsyon Katmanı`
    `x = Conv2D(filters, kernel_size=(3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`

    `# 2. Konvolüsyon Katmanı`
    `x = Conv2D(filters, kernel_size=(3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`

    `#  Skip Connection: Orijinal girişle topluyoruz.`
    `x = Add()([x, shortcut])`
    `x = ReLU()(x)  # Toplamdan sonra aktivasyon fonksiyonu`

    `return x`

# `Model giriş boyutunu belirle`
`inputs = Input(shape=(224, 224, 3))`
`outputs = resnet_basic_block(inputs, 64)`

# `Modeli oluştur`
`model = Model(inputs, outputs)`
`model.summary()`

## **ResNet Bottleneck Block (ResNet-50 ve üzeri modeller için)**

- **3 tane konvolüsyon katmanı kullanır.**
- Girişin boyutunu 1x1 konvolüsyon ile azaltır (verimlilik için).

`def resnet_bottleneck_block(x, filters):`
    `shortcut = Conv2D(filters * 4, kernel_size=(1,1), padding='same')(x)`

    `# 1x1 Konvolüsyon (boyut azaltma)`
    `x = Conv2D(filters, kernel_size=(1,1), padding='same')(x)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`

    `# 3x3 Konvolüsyon`
    `x = Conv2D(filters, kernel_size=(3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`

    `# 1x1 Konvolüsyon (boyut geri büyütme)`
    `x = Conv2D(filters * 4, kernel_size=(1,1), padding='same')(x)`
    `x = BatchNormalization()(x)`

    `# 🔥 Skip Connection: Giriş doğrudan eklenir`
    `x = Add()([x, shortcut])`
    `x = ReLU()(x)`

    `return x`

- İlk olarak, **1x1 konvolüsyon ile girişin boyutu küçültülür.**
- **3x3 konvolüsyon** uygulanarak özellikler öğrenilir.
- **1x1 konvolüsyon ile tekrar boyut büyütülür.**
- **Skip Connection:** Giriş, çıkışa eklenerek gradyan kaybolması önlenir.

## **ResNet Modeli (Tam Model)**

Şimdi **ResNet-50 benzeri bir model oluşturacağız**.

- İlk başta **konvolüsyon + MaxPooling** kullanacağız.
- Daha sonra **resnet bloklarını üst üste koyacağız**.
- En sonda **global average pooling ve tam bağlantılı (fully connected) bir katman ekleyeceğiz**.

`def build_resnet50(input_shape=(224, 224, 3), num_classes=1000):`
    `inputs = Input(shape=input_shape)`

    `# İlk konvolüsyon ve MaxPooling`
    `x = Conv2D(64, kernel_size=(7,7), strides=2, padding='same')(inputs)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`
    `x = tf.keras.layers.MaxPooling2D(pool_size=(3,3), strides=2, padding='same')(x)`

    `# 4 tane residual block (ResNet-50 gibi)`
    `x = resnet_bottleneck_block(x, 64)`
    `x = resnet_bottleneck_block(x, 128)`
    `x = resnet_bottleneck_block(x, 256)`
    `x = resnet_bottleneck_block(x, 512)`

    `# Global Average Pooling + Fully Connected Layer`
    `x = tf.keras.layers.GlobalAveragePooling2D()(x)`
    `x = tf.keras.layers.Dense(num_classes, activation='softmax')(x)`

    `model = Model(inputs, x)`
    `return model`

# `Modeli oluştur`
`model = build_resnet50()`
`model.summary()`



- **İlk konvolüsyon ve MaxPooling:** Modelin girişini işler.
- **4 tane ResNet bloğu eklenir:** **64 → 128 → 256 → 512 filtreli** konvolüsyonlar.
- **Global Average Pooling ve Dense katman:** Sınıflandırma yapılır.

## **Önceden Eğitilmiş ResNet Kullanımı (Keras ile)**

Eğer **sıfırdan bir model eğitmek istemiyorsanız**, Keras’ta hazır eğitilmiş ResNet kullanabilirsiniz.

`from tensorflow.keras.applications import ResNet50`

# `ResNet50 modelini yükleyelim (ImageNet veri setiyle eğitilmiş)`
`resnet_model = ResNet50(weights='imagenet', input_shape=(224, 224, 3))`
`resnet_model.summary()`


# **Xception;**

Xception (Extreme Inception), **Google'dan François Chollet** tarafından önerilen ve **Inception mimarisinin geliştirilmiş hali** olan bir konvolüsyonel sinir ağı (CNN) modelidir. **2017 yılında** yayınlanmıştır.

**Inception Mimarisi**:
**derin öğrenme CNN (Convolutional Neural Network) modelidir**.
CNN mimarilerinde **sabit boyutlu konvolüsyon filtreleri (örneğin 3x3, 5x5) kullanılır**. Ancak farklı ölçeklerdeki nesneleri tanımak için **farklı boyutlarda filtreler gereklidir**.
**Inception mimarisi**, aynı anda **farklı boyutlardaki konvolüsyonları paralel olarak çalıştırarak** daha iyi özellik çıkarımı yapar.

## **Inception Mimarisi Neden Kullanılır?**

1. **Farklı ölçeklerde özellik çıkarımı yapar** → Küçük ve büyük filtreler bir arada kullanılır.
2. **Daha verimli hesaplama sağlar** → **1x1 konvolüsyonlar** kullanılarak hesaplama maliyeti azaltılır.
3. **Derinliği artırırken parametre sayısını düşük tutar** → Daha hızlı ve optimize bir modeldir.
4. **Boyut küçültme ve bilgi kaybını minimize eder** → Daha iyi doğruluk sağlar.

|   |   |   |
|---|---|---|
|**1x1 Konvolüsyon**|Boyutu küçültür, gereksiz bilgiyi atar.|Parametreleri azaltır.|

|   |   |   |
|---|---|---|
|**3x3 Konvolüsyon**|Orta boyutlu özellikleri yakalar.|Ayrıntılı kenar bilgisi çıkarır.|

|                     |                              |                           |
| ------------------- | ---------------------------- | ------------------------- |
| **5x5 Konvolüsyon** | Daha geniş alanları inceler. | Büyük nesneleri tanımlar. |

|   |   |   |
|---|---|---|
|**3x3 Max Pooling**|Görüntüdeki en önemli bölgeleri bulur.|Lokal özellikleri korur.|


Xception, **Inception modüllerini** daha verimli hale getirerek **derin ayrık konvolüsyon (Depthwise Separable Convolution)** kullanır.
 
- **Standart Inception mimarisi**, farklı konvolüsyon filtreleriyle paralel işlemler yaparak verimi artırıyordu.
- **Xception ise**, Inception modülünü **Depthwise Separable Convolution** ile değiştirerek parametre sayısını azaltır ve hesaplama verimini artırır.

## **Xception'in Temel Bileşenleri**

1. **Depthwise Separable Convolution**:
    
    - **Standart Konvolüsyon** işlemini **iki aşamaya** ayırır:
        1. **Depthwise Convolution** → Her kanal için ayrı bir konvolüsyon uygulanır.
        2. **Pointwise Convolution (1x1 Konvolüsyon)** → Kanal sayısını arttırarak özellikleri birleştirir.
   2. **Linear Residual Connections**:
    
    - ResNet’te olduğu gibi, **skip connection** kullanarak **gradyan kaybolmasını önler**.
    3. **Tamamen Ayrıştırılmış Konvolüsyonlar**:
    
    - **Kanal içi ve kanal dışı bilgi akışını birbirinden ayırır**.
    - Bu, Inception modülünden daha iyi bir ayrıştırma sağlar.

## **Xception Mimari Detayları**

Xception, **toplam 36 konvolüsyon katmanı** içerir ve 3 ana bölümden oluşur:

1. **Giriş Konvolüsyon Katmanı (Entry Flow)**
2. **Orta Seviye Katmanlar (Middle Flow)**
3. **Çıkış Katmanı (Exit Flow)**

✔ **Parametreler ve Özellikler**

|Bölüm|Katman Sayısı|Açıklama|
|---|---|---|
|**Entry Flow**|12|İlk özellik çıkarımı, 3 derin ayırıcı konvolüsyon içerir.|
|**Middle Flow**|16|8 tekrar eden bloktan oluşur.|
|**Exit Flow**|8|Son katmanda tam bağlantılı (fully connected) katman ve softmax bulunur.|
`import tensorflow as tf`
`from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, BatchNormalization, Activation`

# Depthwise Seperable Convolution
`def depthwise_separable_conv(x, filters, kernel_size=3, stride=1):`
    `x = DepthwiseConv2D(kernel_size=kernel_size, strides=stride, padding="same", use_bias=False)(x)`
    `x = BatchNormalization()(x)`
    `x = Activation("relu")(x)`
    
    `x = Conv2D(filters, kernel_size=1, padding="same", use_bias=False)(x)`
    `x = BatchNormalization()(x)`
    `x = Activation("relu")(x)`
    
    `return x`


- **DepthwiseConv2D**: Her kanal için ayrı bir konvolüsyon uygular.
- **1x1 Conv2D**: Derinliği genişleterek özellikleri birleştirir.
- **BatchNormalization**: Eğitimi stabilize eder.
- **ReLU Aktivasyonu**: Negatif değerleri sıfırlar.


**Giriş Katmanı EntryFlow;**

`from tensorflow.keras.layers import Input, MaxPooling2D, Add`

`def entry_flow(inputs):`
    `# İlk konvolüsyon bloğu`
    `x = Conv2D(32, kernel_size=3, strides=2, padding="same", use_bias=False)(inputs)`
    `x = BatchNormalization()(x)`
    `x = Activation("relu")(x)`

    `x = Conv2D(64, kernel_size=3, padding="same", use_bias=False)(x)`
    `x = BatchNormalization()(x)`
    `x = Activation("relu")(x)`

    `# 3 Ayrık Konvolüsyon Bloğu`
    `for filters in [128, 256, 728]:`
        `residual = Conv2D(filters, kernel_size=1, strides=2, padding="same", use_bias=False)(x)`
        `residual = BatchNormalization()(residual)`

        `x = depthwise_separable_conv(x, filters)`
        `x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)`

        `x = Add()([x, residual])  # Skip connection ekleniyor`

    `return x`
    
- **İlk 2 konvolüsyon** → Küçük filtrelerle temel özellikleri çıkarır.
- **3 ayrı residual blok** → ResNet gibi giriş özelliklerini derinleştirir.
- **MaxPooling2D** → Boyutu küçültür ve önemli bilgileri korur.
- **Add()** → Residual bağlantıyı ekleyerek gradyan kaybolmasını önler.

**Orta Seviye Katmanlar (Middle Flow)**

`def middle_flow(x):`
    `for _ in range(8):  # 8 tekrar eden blok`
        `residual = x`
        `x = depthwise_separable_conv(x, 728)`
        `x = depthwise_separable_conv(x, 728)`
        `x = depthwise_separable_conv(x, 728)`
        `x = Add()([x, residual])  # Skip connection`
    
    `return x`

- **8 tekrar eden blok** → Modelin derinliğini artırır.
- **Her blokta 3 derin ayrık konvolüsyon** kullanılır.
- **Skip connection (Add)** → Derin öğrenme sürecini stabilize eder.

**Çıkış Katmanı (Exit Flow)**


`def exit_flow(x, num_classes):`
    `residual = Conv2D(1024, kernel_size=1, strides=2, padding="same", use_bias=False)(x)`
    `residual = BatchNormalization()(residual)`

    `x = depthwise_separable_conv(x, 728)`
    `x = depthwise_separable_conv(x, 1024)`
    `x = MaxPooling2D(pool_size=3, strides=2, padding="same")(x)`

    `x = Add()([x, residual])  # Skip connection`

    `x = depthwise_separable_conv(x, 1536)`
    `x = depthwise_separable_conv(x, 2048)`
    `x = tf.keras.layers.GlobalAveragePooling2D()(x)`
    `outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)`

    `return outputs`

- **1024 filtre ile konvolüsyon** → Boyut küçültme ve bilgi yoğunlaştırma.
- **1536 ve 2048 filtreler** → Sonuçları daha detaylı hale getirir.
- **Global Average Pooling** → Boyutu tamamen küçültür.
- **Softmax** → Son sınıflandırmayı yapar.

**Xception Modelini Oluşturma;**

`def Xception(input_shape=(299, 299, 3), num_classes=1000):`
    `inputs = Input(shape=input_shape)`

    `x = entry_flow(inputs)`
    `x = middle_flow(x)`
    `outputs = exit_flow(x, num_classes)`

    `model = tf.keras.Model(inputs, outputs)`
    
    `return model`

# `Modeli oluştur`
`xception_model = Xception()`
`xception_model.summary()`



# **İnception Modeli:**


**İnception Bloğu**

`import tensorflow as tf`
`from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, concatenate, BatchNormalization, Activation`

`def inception_block(x, filters):`
    `f1, f2_in, f2_out, f3_in, f3_out, f4_out = filters`

    `# 1x1 Konvolüsyon`
    `conv1 = Conv2D(f1, (1,1), padding='same', activation='relu')(x)`

    `# 1x1 -> 3x3 Konvolüsyon`
    `conv3 = Conv2D(f2_in, (1,1), padding='same', activation='relu')(x)`
    `conv3 = Conv2D(f2_out, (3,3), padding='same', activation='relu')(conv3)`

    `# 1x1 -> 5x5 Konvolüsyon (Inception v2 ve sonrası yerine 2 adet 3x3 kullanır)`
    `conv5 = Conv2D(f3_in, (1,1), padding='same', activation='relu')(x)`
    `conv5 = Conv2D(f3_out, (3,3), padding='same', activation='relu')(conv5)`

    `# 3x3 MaxPooling + 1x1 Konvolüsyon`
    `pool = MaxPooling2D((3,3), strides=(1,1), padding='same')(x)`
    `pool = Conv2D(f4_out, (1,1), padding='same', activation='relu')(pool)`

    `# Tüm çıktıları birleştir`
    `output = concatenate([conv1, conv3, conv5, pool], axis=-1)`

    `return output`
- **Farklı boyutlarda konvolüsyonlar kullanılır** (1x1, 3x3, 5x5).
- **Pooling katmanı eklenerek daha fazla bilgi çıkarılır.**
- **Tüm sonuçlar birleştirilir (concatenate)**.


# **Inception Modeli (GoogLeNet)**:


`from tensorflow.keras.layers import Input, Flatten, Dense`

`def InceptionV1(input_shape=(224,224,3), num_classes=1000):`
    `inputs = Input(shape=input_shape)`

    `# İlk Konvolüsyon Katmanı`
    `x = Conv2D(64, (7,7), strides=2, padding='same', activation='relu')(inputs)`
    `x = MaxPooling2D((3,3), strides=2, padding='same')(x)`

    `# Inception Bloğu 1`
    `x = inception_block(x, [64, 96, 128, 16, 32, 32])`

    `# Inception Bloğu 2`
    `x = inception_block(x, [128, 128, 192, 32, 96, 64])`

    `# Global Ortalama Havuzlama`
    `x = AveragePooling2D(pool_size=(7,7))(x)`

    `# Tam Bağlantılı Katman ve Çıkış`
    `x = Flatten()(x)`
    `x = Dense(1000, activation='softmax')(x)`

    `model = tf.keras.Model(inputs, x)`
    `return model`

# `Modeli oluştur`
`inception_model = InceptionV1()`
`inception_model.summary()`


- **İlk konvolüsyon katmanı** → Temel özellikleri çıkarır.
- **İki adet Inception bloğu eklenir** → Derinliği artırır.
- **Global Average Pooling ile boyut küçültülür**.
- **Son olarak, softmax katmanı eklenerek sınıflandırma yapılır**.







# Makalede Kullanılan terimler:

**Konvolüsyon**; 
bir görüntü veya sinyal üzerinde belirli bir filtre (kernel) kaydırılarak, yerel özelliklerin (örneğin kenarlar, dokular, desenler) çıkarılmasını sağlar. **giriş verisinden (görüntü, ses, vb.) anlamlı özellikleri otomatik olarak çıkarmayı sağlayan** matematiksel bir işlemdir.

### Detaylı Açıklama:

- **Filtre/Kernel:**  
    Küçük boyutlu (örneğin 3x3, 5x5) bir matristir. Bu filtre, görüntü üzerinde kaydırılır ve her konumda elemanlar arasında çarpma ve toplama işlemleri yapılır.
    
- **Özellik Haritası (Feature Map):**  
    Filtre, giriş görüntüsüne uygulandığında, her konum için bir çıktı değeri üretir. Tüm bu değerler bir araya gelerek yeni bir **özellik haritası** oluşturur. Bu harita, orijinal görüntünün belirli özelliklerini vurgular.
    
- **Örnek:**  
    Eğer bir kenar algılama filtresi kullanılırsa, bu işlem sonucunda kenarların bulunduğu bölgeler yüksek değerler verirken, diğer bölgeler düşük değerler üretecektir.
    
- **Kullanım Alanı:**  
    Konvolüsyon işlemi, görüntülerdeki düşük seviyeli özelliklerin (örneğin kenarlar, köşeler, renk geçişleri) çıkarılmasında kritik rol oynar. Bu bilgiler daha sonra, daha derin katmanlarda daha karmaşık ve soyut özelliklerin öğrenilmesinde kullanılır.



**Skip Connection(Atlama Bağlantısı);**
bir derin sinir ağı (DNN) içinde **önceki katmandaki çıktıyı (X) birkaç katman sonrasına doğrudan ekleyen** bir mekanizmadır.

Bu yapı, özellikle **ResNet (Residual Network)** gibi **çok derin ağlarda** kullanılır ve **gradyan kaybolma (vanishing gradient)** problemini önlemeye yardımcı olur.

**Gradyan Kaybolma (Vanishing Gradient) Problemi**;
- Derin ağlarda, geri yayılım (backpropagation) sırasında gradyanlar **sıfıra yakınsayarak küçülebilir**.
- Küçük gradyanlar, **erken katmanların öğrenmesini engeller** ve eğitim zorlaşır.
- **Çözüm:** Skip Connection kullanarak **gradyanların doğrudan erken katmanlara akmasını sağlarız**.

- **Derin katmanlar**, verinin orijinal şeklini değiştirdiğinden **bazı özellikler kaybolabilir**.
- **Skip Connection, orijinal bilgiyi doğrudan sonraki katmanlara ileterek** bunu önler.
- Geleneksel ağlarda, modelin **f(x) gibi karmaşık bir fonksiyonu öğrenmesi gerekir**.
- **Skip Connection eklenirse, model yalnızca "g(x) = f(x) - x" fonksiyonunu öğrenir.**
- **Bu, öğrenmeyi hızlandırır ve hata oranını düşürür.**

**Skip Connection Türleri;**
### **Basit Residual (Kalıntı) Bağlantısı**

Bu en yaygın kullanılan türdür ve **ResNet** gibi mimarilerde görülür.

Matematiksel gösterimi:
	$y=f(x)+x$

- **f(x)** → Katmanların öğrenmeye çalıştığı fonksiyon.
- **x** → Önceki katmandan gelen giriş, doğrudan çıkışa eklenir.

**Resnet Block yapısı**


`import tensorflow as tf`
`from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, Add, Input`

`def residual_block(x, filters):`
    `shortcut = x  # Skip connection için orijinal giriş`

    `# 1. Konvolüsyon Katmanı`
    `x = Conv2D(filters, (3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`

    `# 2. Konvolüsyon Katmanı`
    `x = Conv2D(filters, (3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`

    `# Skip Connection: Giriş doğrudan çıkışa eklenir`
    `x = Add()([x, shortcut])`
    `x = ReLU()(x)`

    `return x`

# `Model giriş boyutunu belirle`
`inputs = Input(shape=(224, 224, 3))`
`outputs = residual_block(inputs, 64)`

# `Modeli oluştur`
`model = tf.keras.Model(inputs, outputs)`
`model.summary()`


- İlk olarak **3x3 konvolüsyon** ve aktivasyon uygulanır.
- İkinci konvolüsyon uygulanır.
- **Giriş (shortcut) doğrudan çıkışa eklenerek skip connection oluşturulur.**

### **Projection Shortcut (Boyut Eşleme)**

Eğer giriş ve çıkışın **boyutları aynı değilse**, doğrudan toplama işlemi yapılamaz.  
Bunu çözmek için, **1x1 konvolüsyon ile girişin boyutu değiştirilir** ve sonra ekleme yapılır:

Matematiksel gösterimi:
$y=f(x)+Ws​⋅x$

**W_s**, girişin boyutunu eşleyen **1x1 konvolüsyon filtresidir**.



`def projection_residual_block(x, filters):`
    `# Girişin boyutunu değiştiren 1x1 konvolüsyon`
    `shortcut = Conv2D(filters, (1,1), strides=1, padding='same')(x)`
    
    `# 3x3 konvolüsyon katmanları`
    `x = Conv2D(filters, (3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`
    `x = ReLU()(x)`
    
    `x = Conv2D(filters, (3,3), padding='same')(x)`
    `x = BatchNormalization()(x)`
    
    `# Skip Connection (1x1 konvolüsyon ile boyut eşleme)`
    `x = Add()([x, shortcut])`
    `x = ReLU()(x)`

    `return x`

- **1x1 konvolüsyon, girişin boyutunu değiştirir.**
- Boyutu değişen giriş, normal çıkış ile toplanır.

### **Dense Connection (Yoğun Bağlantı) – DenseNet**

- **Her katmandan gelen çıkış, tüm sonraki katmanlara bağlanır.**
- ResNet'ten farklı olarak, sadece önceki katmana değil, **tüm önceki katmanlara bağlantı ekler.**
- **Daha fazla bilgi paylaşımı sağlar ve daha verimli öğrenme sunar.**

 **Matematiksel gösterimi:**
$y=[x1​,x2​,x3​,...,xn​]$

Her katmanın çıktısı, tüm sonraki katmanlarla birleştirilir.
