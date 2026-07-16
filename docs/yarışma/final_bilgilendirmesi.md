FİNAL AŞAMASI: Değerlendirme Süreci ve Çözümlerin İletim Formatı

Yarışma bitmeden önce finale taşımak istediğiniz iki adet submission'unuzu manuel olarak seçmenizi öneriyoruz. Otomatik olarak seçilen ama çözümünü yollayamayacağınız, kurallara aykırı olan bir submissionun best Private LB skorunuz gelmesi halinde maalesef finale geçme hakkınızı kaybedebilirsiniz. Her takımdan seçtikleri iki submission'un sadece Private LB'de en iyi sonucu vermiş olanına ait çözümü istiyor olacağız.

Yarışmanın sonucunda Private LB'deki top 20 finalist adayı olan takım, çözümlerinin doğrulanabilmesi ve kontrol edilebilmesi için aşağıdaki dosyaları içeren sadece bir adet Private Kaggle dataset oluşturup trendyoldatascience ve coderspacetr kullanıcılarına erişim vermelidir. Datasetin ismi TAKIMISMI_Trendyol2026_Solution formatında olmalıdır.

Finalist adaylarının:

    18 Temmuz 2026 00:00 – 25 Temmuz 2026 23:59 arasında çözümlerini iletmiş olmaları gerekmektedir.
    18 Temmuz 2026 00:00 – 1 Ağustos 2026 23:59 arasında ekibimiz yollanan çözümlerin kontrolünü yapacak ve gerekli durumlarda (çözümlerin yanlış veya eksik çalışması vb.) takımlar ile ara ara iletişime geçecektir.

Çözüm kontrolünü geçen en yüksek Private LB skoruna sahip Top10 takım finalist olarak sonraki aşamaya geçecektir.

Takımların ilettiği datasetin içerisinde:

    Çözüme ait:

        Üretilen bütün ek veriler, sentetik veriler, pseudo-labeled veriler ve varsa negatif örnekler

        Veri hazırlama kodu

        Negatif örnek, sentetik örnek, pseudo-label üretiliyorsa o da dahil yarışma verisi haricinde kullandığınız bütün verilerin üretim süreçlerini iletmeniz gerekiyor. Ek verileriniz ile onu üreten rutinleri eş isimlendirmenizi rica ediyoruz. Aşağıdaki gibi bir formatı tercih edebilirsiniz:
            generate_external_data_llm.py , external_data_llm.csv
            generate_external_data_synth.py , external_data_synth.csv
            …

        Eğitim - validasyon kodu

        Inference kodu (Bu kod internet kullanmadan çalışabilmelidir)

    Eğitilen modeller (Büyük modellere LORA finetune yapmak gibi durumlarda orjinal backbone modele dokunmadığınız için upload edeceğiniz datasette de yüklemenize gerek olmayacaktır (eğer ilgili baz modellerin kaynaklarından silinmeyeceğine eminseniz, silinme riski varsa yine de upload etmenizi istiyoruz. modele ulaşamamamız durumunda doğrulama da yapamayacağımız için çözümünüz geçersiz sayılabilir). Sadece eğittiğiniz modelleri upload ettiğiniz, geri kalan baz model parçalarının ortam hazırlama aşamasında otomatik olarak indirildiği bir yapı kurgulayınız.)

    Çözümün geliştirildiği makinenin donanım bilgileri (GPU marka-model-adet, RAM miktarı, CPU marka-model)

    Çözümün bütün aşamalarının çalıştırılabilmesi için gereken environment şablonu (conda env yaml, pip requirements txt vs.)

    Çözümün bütün aşamalarının çalıştırılabilmesi için indirilmesi veya yüklenmesi gereken ek yazılım-model-kütüphanelerin bir listesi ve kurulumlarına ait rehber

    Çözüme ait aşağıdaki üç ana akışı uçtan uca çalıştıracak rutinler oluşturmanızı istiyoruz, birden fazla bağımsız scripti ardışık çalıştıran bash-scriptler olarak da tasarlayabilirsiniz:
        Ortam oluşturma + ortamı aktif etme + gereksinimlerin, backbone modellerin, gereken bütün framework-driver-kütüphanelerin indirilip yüklenmesi (örnek olarak adına step1.sh diyelim)
        Ortamı aktif etme + veri hazırlama & üretme + eğitim + model kaydetme (örnek olarak adına step2.sh diyelim)
        Bu aşamada çalışan bütün ek veri üretim scriptleri ortak bir klasöre ayrı dosyalar olarak çıktı üretmelidir. 'Veri hazırlama kodu' bölümünde verdiğimiz örneğe göre: extra_generated_data/ dizini altına external_data_llm.csv, external_data_synth.csv dosyalarının kaydedilmesi gibi düşünebilirsiniz.
        Bu aşamada çalışan bütün eğitim scriptleri ortak bir klasöre ayrı dosyalar olarak çıktı üretmelidir. models/ dizini altına model1.pth, model2.pickle dosyalarının kaydedilmesi gibi düşünebilirsiniz.
        Bu script aşağıdaki parametreleri girdi olarak almalıdır:
            competition_data_path: Yarışma verisinin bulunduğu path (örneğimize göre competition_data/)
            extra_data_path: Ek üretilecek verinin kaydedileceği path (örneğimize göre extra_generated_data/)
            model_dump_path: Eğitilen modellerin kaydedileceği path (örneğimize göre models/)
        Ortamı aktif etme + model yükleme + tahmin (internetsiz çalışabilmeli) (örnek olarak adına step3.sh diyelim)
        Bu script aşağıdaki parametreleri girdi olarak almalıdır:
            model_dump_path: Inference aşamasında kullanılacak modellerin path'i
            competition_data_path: Yarışma verisinin bulunduğu path, submission pairs dosyasını otomatik buluyor olacaksınız (örneğimize göre competition_data/)
            out_path: Sonucun yazılacağı path
        Bu aşama aldığı model pathi ile gerekli model yüklemelerini yapmalı ve verilen test dosyasında koşarak ürettiği cevapları istenen adrese yazmalıdır.

olmalıdır.

Yukarıda anlattığımız şemaya göre bize yollanan bir çözümün aşağıdaki aşamalarla test edileceğini söyleyebiliriz:

    İster dosyaların kontrolü
        Çözüm talep ettiğimiz bütün süreçlere ait kodları içeriyor mu?
        Ek veri üretiminde test verisinde koşan bir ücretli servis var mı?
        Üretildiği söylenen her ek verinin üretim scripti verilmiş mi?
        Veri üretim scriptlerinizi koşup aynı verinin üretilmesi gibi bir beklentimiz rastlantısallıktan dolayı olmayacak. Fakat ek veri olduğunu beyan ettiğiniz çıktıların test setine benzerliğini, çıktıyı üreten scriptin çalışma biçimini detaylı olarak hem takım üyelerimiz hem de otonom araçlar inceliyor olacak.
        Çözüm ortamına ait donanım detayları verilmiş mi?
        Talep edilen üç aşamalı script yapısı sağlanmış mı?
        Tahmin aşamasında kullanılan bütün modeller iletilmiş mi?

    Çözümün kontrolü
        Beyan edilen donanım özelliklerine yakın bir makine kaldırılır
        Gerekli ortamın kurulumunu sağlayan script çalıştırılır Örn: (bash step1.sh)
        İnternet erişimi kapatılır
        Tahmin modülü takımın ilettiği modeller ile çalıştırılır Örn: (bash step3.sh --model_dump_path models/ --competition_data_path competition_data/ --out_path submission.csv)
        Çözümün koştuğu teyit edilir.
        Çözümün sıralamada gözüken Private LB skoru ile aynı - çok yakın skor ürettiği teyit edilir. Bu kontroldeki yakınlık eşiğimiz +- 0.003 olacak.
        Veri üretimi ve model eğitimlerini sağlayan script çalıştırılır Örn: (bash step2.sh --competition_data_path competition_data/ --extra_data_path extra_generated_data/ --model_dump_path new_trained_models/ )
        Kullanıldığı iddia edilen bütün ek verilerin yeniden, belirtilen path altında oluşturulduğu ve oluşan dosya isimlerinin oluşturan script isimleri ile uyumluluğu teyit edilir.
        Eğitimlerin hatasız tamamlandığı ve belirtilen klasöre ilk aşamada denenen yapı ile aynı yapıda kaydettiği teyit edilir.
        Tahmin modülü yeni eğitilen modeller ile çalıştırılır Örn: (bash step3.sh --model_dump_path new_trained_models/ --competition_data_path competition_data/ --out_path submission.csv)
        Yeniden eğitilmiş çözümün inference koştuğu teyit edilir.
        Yeniden eğitilmiş çözümün ilk alınan inference ile çok yüksek benzerlik taşıdığı teyit edilir.
        Yeniden eğitilmiş çözümün sıralamada gözüken Private LB skoru ile çok yakın skor ürettiği teyit edilir. Bu kontroldeki yakınlık eşiğimiz +- 0.005 olacak. Çözümün içerdiği rastlantısal süreçlere göre esneklik tanıyacağız.
        Bu kontrol aşamalarından bağımsız olarak kontrolü geçen bütün takımlar yine de görünen Kaggle Private Leaderboard skorları üzerinden değerlendirilmiş olacaklar.

Dikkat edilmesi gereken hususlar:

    Çözümün yarışma test setinde tahminleme yapan bütün aşamaları internet kullanmadan çalışabilmelidir.

    Birden fazla çözümü beraber (ensemble) kullanmanız durumunda iletilen dosyanın kullanılan bütün çözümleri içermesi beklenmektedir, ilgili private LB skorunu alan rutininiz birebir bütün çözümleri aynı şekilde, aynı sırada çalıştırmalı ve birleştirmeli.

    Çözümünüz birden fazla çözümü sıralı (stacking) halde kullanıyorsa iletilen çözümün yalnızca son aşamayı değil, yine deneyin başından beri çalıştırılmış tüm aşamaları içermesi beklenmektedir, ilgili private LB skorunu alan rutininiz birebir bütün çözümleri aynı şekilde, aynı sırada çalıştırmalı ve birleştirmeli.

    İlettiğiniz dosyalardaki veri hazırlama + eğitim kodları çalıştırıldığında yolladığınız modeller ile yakın çıktılar veren modellerin eğitilebilmesi beklenmektedir.

    Yolladığınız çözümün inference kodu yarışma test seti üzerinde ilettiğiniz model ile çalıştırıldığında Kaggle Private LB skorunuza çok yakın bir sonuç alınması beklenmektedir. Bunun gerçekleşmemesi ve haklı bir gerekçesi olmaması durumunda diskalafiye olabilirsiniz. Seçtiğiniz submission dışında başka bir submissionu reproduce eden bir çözüm yollamanız halinde; çözümünüz seçtiğiniz submission ile bahsettiğimiz +- 0.003 sınırlarında sonuç vermiyorsa finale maalesef geçemiyor olacaksınız. Seçilen submission ve iletilen çözümün uyumu hakkında katılımcılardan tamamen dürüst olunmasını bekliyoruz.

    Çözümünüzün herhangi bir aşamasında (sentetik-negatif-pseudolabeled veri üretimi, eğitim, inference vb.) generative olarak lokal LLM kullanıyorsanız lütfen promptlarını ve sampling parametrelerini de ayarlı bir şekilde iletmeyi ihmal etmeyin. Bu süreçler çalıştırıldığında LLM’inizin ürettiği çıktıların ilettiğiniz ek veriler ile yüksek benzerlikte olmasını bekliyoruz. Bütün veri üretim süreçlerini koşup bütün dosyalarda benzerlik amaçlamıyoruz, fakat ufak bir sette uyumlu olmasını da bekleriz.

    Çözümlerin herhangi bir tekrarlanabilirlik veya çalışma aşamasında sorun yaşanması durumunda takımlarla iletişime geçiyor olacağız. Suistimal olmayan bütün problemlerde olabildiğince yardımcı olmaya çalışacağız.