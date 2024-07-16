# hupunct
I aim to fill the gap between Speech Recognition (speech2text) and downstream NLP tasks by developing a model for Automatic Punctuation Restoration (APR) in Hungarian called ‘hupunct’, that has raw unpunctuated lower-cased text as its input, and has the corrected, punctuated text as its output. The solution is based on a widely used NLP technique, which involves the finetuning of a pretrained special deep neural network, a Transformer. The hupunct model, after training for less than one epoch on the dataset generated from the Hungarian Web Corpus reached a test micro average F1-score of 87.2% and macro average F1-score of 74,1%. The CDQ macro F1-score achieved was 83.7%. This surpasses the current state-of-the art Hungarian model, although on a different but arguably harder dataset, even with using only one prediction per token. The model learned to restore punctuations belonging to the additional base punctuation classes and all the upper versions of those classes to a reasonable extent. Additionally, it can also auto-capitalize, which is a convenient feature. The finetuning of huBERT for the APR task in Hungarian proved to be a powerful and very practical approach, especially with the usage of the HF platform.

### Examples
Input: 'gerendai páltól a következőt idézzük gyermekkorom óta szeretem a balatont a balatoni tájak mindig is lenyűgöztek és néha néha mikor a balaton partján sétálok szívemet elönti a szeretet hogyan lehet valami ilyen szép a következő vendégünk hambuch kevin a balatonfenyvesi egyetem doktora a knorr bremse kutatás fejlesztésért felelős vezetője kevin ilyen olyan projektekben vett részt a mta val közösen majd 1999 ben alapítottak barátjával csisztapusztai arnolddal egy céget megpedíg a gránit kft t ezután kezdte meg tevékenységét a német cégnél ahol a gránit kft ben szerzett tapasztalatát kamatoztatja'

Output: 'Gerendai Páltól a következőt idézzük: Gyermekkorom óta szeretem a Balatont. A balatoni tájak mindig is lenyűgöztek, és néha-néha, mikor a Balaton partján sétálok, szívemet elönti a szeretet. Hogyan lehet valami ilyen szép? A következő vendégünk Hambuch Kevin, a Balatonfenyvesi Egyetem doktora, a Knorr-Bremse kutatás-fejlesztésért felelős vezetője. Kevin ilyen-olyan projektekben vett részt a Mta-val közösen, majd 1999-ben alapítottak barátjával, Csisztapusztai Arnolddal egy céget, megpedíg a Gránit Kft-t. Ezután kezdte meg tevékenységét a német cégnél, ahol a Gránit Kft-ben szerzett tapasztalatát kamatoztatja.'

Developed by: Tamás Gyenis - tamgyen@gmail.com
