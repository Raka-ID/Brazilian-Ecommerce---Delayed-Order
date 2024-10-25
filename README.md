# Analisis Brazillian E-commerce (Olist) Delay Order
[![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)

## Summary
Proyek ini bertujuan untuk menganalisis dan memprediksi _delay_ pengiriman _order_ pada platform _e-commerce_ di Brasil bernama **Olist**. Dengan pertumbuhan _e-commerce_ yang pesat, kepuasan _customer_, terutama terkait waktu pengiriman, sangat penting untuk keberhasilan bisnis. Analisis ini bertujuan untuk memahami masalah bisnis terkait _delay_ pengiriman, dampaknya terhadap kepuasan _customer_, dan membuat model machine learning untuk memprediksi risiko _delay_ di masa mendatang.

## Business Understanding
Persaingan yang semakin ketat dalam _e-commerce_ memaksa _seller_ untuk memastikan pengiriman yang efisien. _Delay_ pengiriman dapat menyebabkan ulasan negatif dari _customer_, yang pada akhirnya memengaruhi kepuasan _customer_, reputasi, dan penjualan. _Project_ ini berfokus pada pemahaman isu-isu berikut:
1. Kerusakan Reputasi
2. Pengurangan Penjualan
3. Kehilangan _customer_ (_Customer Churn_)
4. Peningkatan Biaya Dukungan _customer_
5. Kerusakan Hubungan antara Platform _E-commerce_ dan _seller_
6. Dampak pada Algoritma dan Visibilitas Produk
7. Potensi _Refund_ atau _Return_

Sebuah simulasi hipotetis diberikan untuk menghitung potensi kerugian finansial bulanan akibat _delay_ pengiriman dan ulasan negatif berikutnya, yang berjumlah sekitar **$10.013,33**.

## Dataset
**Dataset ini disediakan oleh Olist**, sebuah department store terbesar di pasar e-commerce Brasil ([kaggle](https://www.kaggle.com/datasets/olistbr/brazilian-ecommerce?select=olist_sellers_dataset.csv)). Dataset yang digunakan dalam analisis ini mencakup detail pesanan, ulasan pelanggan, dan informasi pengiriman dari platform e-commerce Brasil, berupa bentuk tabel-tabel terpisah. Data yang digunakan merupakan kumpulan data order dari Januari 2017 hingga Agustus 2018. Fitur-fitur yang terdapat dalam dataset ini meliputi:

- ID pesanan, ID pelanggan, Kategori Produk
- Tanggal Pengiriman yang Diperkirakan dan Tanggal Pengiriman Aktual
- Skor Ulasan dan Review Customer
- Detail _seller_ dan Informasi Logistik

**Deskripsi Tabel**

| Nama Kolom | Tabel Asal | Deskripsi | Keterangan |
|---|---|---|---|
|`order_id`|order|Kode unik untuk menunjukan nomor id order dari Olist|-|
|`customer_id`|order|Kode unik untuk setiap _customer_ pada tiap `order_id`|Lebih detail di tabel _customer_|
|`order_status`|order|Status dari order yang telah dibuat|'delivered', 'invoiced', 'shipped', 'processing',<br>'unavailable', 'canceled', 'created', 'approved'|
|`order_purchase_timestamp`|order|Waktu saat order dibeli|Format: YYYY-MM-DD HH:MM:SS|
|`order_approved_at`|order|Waktu saat order disetujui|Format: YYYY-MM-DD HH:MM:SS|
|`order_delivered_carrier_date`|order|Tanggal saat order diserahkan kepada kurir|Format: YYYY-MM-DD HH:MM:SS|
|`order_delivered_customer_date`|order|Tanggal saat order diterima oleh _customer_|Format: YYYY-MM-DD HH:MM:SS|
|`order_estimated_delivery_date`|order|Tanggal estimasi pengiriman order|Format: YYYY-MM-DD HH:MM:SS|
| `customer_id` |customer| ID unik _customer_ yang terhubung dengan tabel order | Berupa angka atau kode unik |
| `customer_unique_id` |customer| ID unik yang mewakili setiap _customer_ secara individu  | - |
| `customer_zip_code_prefix` |customer| Kode pos tempat tinggal _customer_ | - |
| `customer_city` |customer| Nama kota tempat tinggal _customer_ | Lokasi kota _customer_ tinggal |
| `customer_state` |customer| Kode negara bagian tempat tinggal _customer_ | Contoh: SP untuk São Paulo di Brasil |
| `order_id` |item| Kode unik untuk menunjukan nomor id order dari Olist|-|
| `order_item_id` |item| ID unik item dalam order | Mengidentifikasi setiap item dalam order |
| `product_id` |item| ID unik produk | Mengidentifikasi produk yang dibeli|
| `seller_id` |item| ID unik _seller_ | Terhubung ke identitas _seller_|
| `shipping_limit_date` |item| Batas waktu pengiriman | Tanggal terakhir pengiriman harus dilakukan |
| `price` |item| Harga produk | Harga per item |
| `freight_value` |item| Biaya pengiriman | Ongkos kirim yang dikenakan|
| `product_id`|produk| ID unik produk| Mengidentifikasi produk yang dibeli|
| `product_category_name`|produk| Nama kategori produk| - |
| `product_name_lenght`|produk| Panjang nama produk dalam karakter| - |
| `product_description_lenght`|produk| Panjang deskripsi produk dalam karakter|  -|
| `product_photos_qty`|produk| Jumlah foto produk yang disertakan|- |
| `product_weight_g`|produk| Berat produk | dalam satuan gram|
| `product_length_cm`|produk| Panjang produk |dalam sentimeter|
| `product_height_cm`|produk| Tinggi produk|dalam sentimeter|
| `product_width_cm`|produk| Lebar produk| dalam sentimeter |
| `seller_id`|seller| ID unik _seller_| Mengidentifikasi _seller_|
| `seller_zip_code_prefix`|seller| Kode pos tempat tinggal _seller_| -|
| `seller_city`|seller| Nama kota tempat tinggal _seller_| -|
| `seller_state`|seller| Kode negara bagian tempat tinggal _seller_| Contoh: SP untuk Sao Paulo di Brasil|
| `order_id`|payment| Kode unik untuk menunjukan nomor id order dari Olist|-|
| `payment_sequential`|payment| Urutan pembayaran terkait dengan order| - |
| `payment_type`|payment| Jenis metode pembayaran yang digunakan| Contoh: credit_card, boleto, voucher, dll. |
| `payment_installments`|payment| Jumlah cicilan yang dilakukan untuk pembayaran order| -|
| `payment_value`|payment| Nilai total yang dibayarkan untuk order| - |
| `geolocation_zip_code_prefix`|geolocation| Kkode pos lokasi| -|
| `geolocation_lat`|geolocation| Garis lintang lokasi (latitude)| Koordinat geografis|
| `geolocation_lng`|geolocation| Garis bujur lokasi (longitude)| Koordinat geografis|
| `geolocation_city`|geolocation| Nama kota berdasarkan lokasi geolokasi|- |
| `geolocation_state`|geolocation| Kode negara bagian berdasarkan lokasi geolokasi| Contoh: SP untuk São Paulo di Brasil|
| `product_category_name`|category translation| Nama kategori produk dalam bahasa asli| Portugis |
| `product_category_name_english` |category translation| Nama kategori produk dalam bahasa terjemahan| Inggris |

![Mind Map](https://i.imgur.com/HRhd2Y0.png)

## Project Structure
1. **Pemahaman Masalah Bisnis**: Eksplorasi mendetail tentang konteks, dampak, dan kerugian finansial akibat _delay_ pengiriman.
2. **Eksplorasi dan Pra-pemrosesan Data**: Memuat dataset, membersihkan, dan mengeksplorasi data untuk memahami variabel-variabel dan tren utama.
3. **Feature Engineering**: Membuat fitur baru seperti deviasi waktu pengiriman, sentimen ulasan, dan analisis mitra logistik untuk meningkatkan daya prediksi.
4. **Pemodelan**: Membangun model machine learning untuk memprediksi apakah suatu _order_ akan terlambat atau tepat waktu. Beberapa model diuji, dan model terbaik dipilih berdasarkan metrik performa seperti RMSE, MAE, MAPE.
5. **Evaluasi dan Insight**: Mengevaluasi performa model, menginterpretasikan hasil, dan memberikan insight yang dapat ditindaklanjuti untuk platform _e-commerce_ guna meminimalkan _delay_ dan meningkatkan kepuasan _customer_.

## Results
Berdasarkan analisis yang dilakukan terhadap keterlambatan pengiriman dan dampaknya terhadap ulasan negatif, berikut adalah hasil utama yang diperoleh dari proyek ini:

1. **Volume Pesanan yang Melonjak Selama Periode Sibuk**: Lonjakan volume pesanan yang signifikan, terutama selama periode sibuk seperti hari libur nasional dan _Black Friday_, berkontribusi pada peningkatan keterlambatan pengiriman. Hal ini menunjukkan perlunya strategi khusus dalam menghadapi periode dengan volume pesanan tinggi.

2. **Keterlambatan pada Tahap Akhir Pengiriman**: Keterlambatan pengiriman terutama terjadi pada tahap akhir, yaitu ketika pesanan sudah diterima oleh kurir dan sedang dalam proses pengantaran kepada pelanggan. Ini menunjukkan bahwa fase last-mile delivery memerlukan perhatian lebih dalam optimasi proses logistik.

3. **Dampak Negatif terhadap Reputasi dan Kepercayaan Pelanggan**: Keterlambatan pengiriman mengakibatkan ulasan negatif yang berdampak langsung pada reputasi platform e-commerce. Hal ini berpotensi mengurangi kepercayaan pelanggan dan mempengaruhi penjualan jangka panjang.

4. **Pengiriman di Lokasi Terpencil**: Lokasi-lokasi terpencil, baik di tingkat kota, negara bagian, maupun rute tertentu, memiliki waktu pengiriman yang lebih lama. Hal ini menyoroti pentingnya perencanaan logistik yang lebih efektif untuk daerah-daerah yang cenderung memiliki risiko keterlambatan lebih tinggi.

5. **Pengaruh Volume Pesanan pada Tingkat Keterlambatan**: Kenaikan volume pesanan pada tanggal-tanggal tertentu seperti hari libur atau event penjualan besar berhubungan erat dengan peningkatan tingkat keterlambatan pengiriman, mengindikasikan kebutuhan perencanaan logistik dan pengelolaan sumber daya yang lebih baik selama periode tersebut.

6. **Prediksi dengan Model Time Series SARIMAX dan Gradient Boost**: Model time series SARIMAX digunakan untuk memprediksi total order dengan RMSE = 66.90, sementara model Gradient Boost digunakan untuk memprediksi jumlah keterlambatan dengan RMSE = 7.59. Hasil ini menunjukkan bahwa prediksi menggunakan dua pendekatan ini dapat memberikan gambaran yang jelas tentang tren dan risiko keterlambatan.

7. **Prediksi Keterlambatan Empat Bulan ke Depan**: Hasil prediksi menunjukkan bahwa total order yang diprediksi untuk bulan Agustus adalah 6.756 dengan jumlah keterlambatan 237, bulan September sebanyak 6.442 dengan keterlambatan 411, bulan Oktober sebanyak 6.523 dengan keterlambatan 233, dan bulan November sebanyak 6.180 dengan keterlambatan 601. Tren ini menunjukkan total pesanan yang stagnan tetapi dengan peningkatan jumlah keterlambatan.

8. **Kerugian Finansial Akibat Keterlambatan**: Kerugian yang diprediksi akibat keterlambatan pengiriman dalam empat bulan ke depan adalah $7.608,83 untuk Agustus, $8.645,75 untuk September, $9.539,64 untuk Oktober, dan $9.856,47 untuk November. Kerugian ini menunjukkan dampak finansial signifikan yang perlu ditangani dengan strategi mitigasi yang tepat.

## Dashboard
[**Dashboard Tableau**](https://public.tableau.com/views/E-CommerceDashboardDelayDeliveryCase/Page2?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link) dibuat untuk visualisasi alternatif hasil analisis ini. Bagi yang tertarik dapat melihat visualisasi tersebut untuk mendapatkan wawasan lebih mendalam mengenai keterlambatan pengiriman dan dampaknya. Dashboard ini tidak hanya memberikan visualisasi yang mudah dipahami mengenai tren keterlambatan dan volume pesanan, tetapi juga membantu mengidentifikasi area-area kritis yang memerlukan perhatian lebih.

## Tools and Libraries
- **Python**: untuk analisis data dan pemodelan
- **Jupyter Notebook**: untuk pengembangan kode interaktif
- **Pandas, NumPy**: untuk manipulasi dan analisis data
- **Scikit-Learn**: untuk pemodelan machine learning
- **Matplotlib, Seaborn**: untuk visualisasi data

## Getting Started
Untuk menjalankan analisis ini secara lokal:
1. Clone repositori ini.
2. Install paket-paket yang diperlukan dengan `pip install -r requirements.txt`.
3. Buka Jupyter Notebook dan jalankan secara berurutan.

## Conclusion
_Delay_ pengiriman berdampak signifikan pada kepuasan _customer_ dan reputasi keseluruhan platform _e-commerce_. Dengan memprediksi kemungkinan _delay_ pengiriman, platform dapat mengambil langkah-langkah proaktif untuk mengurangi dampaknya terhadap _customer_ dan meningkatkan kualitas layanan.

## Future Work
- Integrasi data pelacakan pengiriman secara real-time untuk meningkatkan akurasi prediksi.
- Mengembangkan sistem rekomendasi untuk mitra logistik berdasarkan kinerja historis.
- Mengimplementasikan A/B testing untuk mengevaluasi efektivitas notifikasi proaktif kepada _customer_ terkait _delay_ yang diprediksi.

## Author
Proyek ini dibuat oleh Tim Alpha sebagai bagian dari upaya untuk meningkatkan logistik _e-commerce_ dan pengalaman _customer_.
