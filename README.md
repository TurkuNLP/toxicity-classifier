# toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), multiclass/binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is available in [huggingface](https://huggingface.co/datasets/TurkuNLP/wikipedia-toxicity-data-fi) and a model trained on this data is available [here](https://huggingface.co/TurkuNLP/bert-large-finnish-cased-toxicity). 

New annotated data for Finnish based on a sample from Suomi24 is available in the folder annotations in the file all_annotations.tsv and in [huggingface](https://huggingface.co/datasets/TurkuNLP/Suomi24-toxicity-annotated). The script for evaluating that data is in the folder predictions.


## NOTE THAT THERE ARE EXPLICIT EXAMPLES FROM THE TEST DATA BELOW



## Examples of mistakes the model makes for the test set

| ID | Text | Gold label | predicted label |
| -- | ---- | ---------- | --------------- |
| 166f8512c037f17a | HELVETTIIN, PUTIN, SENKIN MURHAAVA PSYKOPAATTI! | insult, toxicity | insult, obscene, severe_toxicity, toxicity |
| e4d80d9fa7e13eed | \" \n ::::::Kiitos ajastanne.  Tämän käyttäjän ensimmäinen muokkaus on yhtä kuin jättimäinen \"\"Haista vittu Wikipedia, trollaan sinua niin kovaa, että tarkistat tarkistajat, kun olen valmis.\"\". Banskyn kuvan lisääminen käyttäjäsivulle ja käyttäjänimen valinta puhuvat kirjaimellisesti asiaa.   \" | - | obscene, toxicity |
| 4d27b01205fd3dc7 | চাপিতলা গ্রামটি একটি প্রাচীন গ্রাম। এই গ্রামটির বয়স প্রায় ৫০০ বছর হবে। এই গ্রামে অনেক কৃতি সন্তানের জন্ম হয়েছে। তাদের মদ্দে অন্নতম হল আলহাজ আব্দুল হাসেম সাব। এই গ্রামটি। এই গ্রামটি। বর্তমানে মুরাদ নাগার থানার একটি বড় ইউনিয়ন এই ইউনিয়ন এর লক সংখ্যা প্রায় ৫০০০০ এর বেশী। এই গ্রামে গ্রামে ১ টি মাদ্দমিক বিদ্যালয় ১ টি কলেজ এবং ৫ টি প্রাইমারী বিদ্যালয় আছে। | - | toxicity |
| 943eec39cc188f33 | ::* Corinne olet liian fiksu. | - | insult, toxicity |
| 805a962e6639dff7 | \" \n ::: xD Katsoin juuri eBaysta, parempi ostaa se hänen sivustoltaan.... Joku ämmä myy deluxe debyyttiä 400$. o.O ℥- \" | - | obscene, toxicity |
| 05f7b43deb23555f | Kyllä, ja Davie on homoseksuaali... | - | identity_attack, toxicity |
| eb79c0092ce7ec64 | "== Ok == \n\n Testasin vain, kuinka nopeasti typeriin muokkauksiin reagoidaan... \n\n  Se tyyppi, jonka juuri hoidit. | toxicity | - |
| 3d9d62c203b619ed | == One for you == \n\n En tiedä, ylläpidätkö vielä muokkausluetteloasi, mutta tämä muokkaus - ja sitä seurannut selitys - on varmasti aivan Wikipedian typeryyksien listalla. | insult, toxicity | toxicity |
| ec2f87887d66f638 | kukko liha hiekka joka mdjdht | - | insult, obscene, toxicity |
| 6adbbabff7afbb5d | homo, joka tykkää mulkusta perseessään. Hän juoksi niin paljon kiristääkseen persettään | insult, obscene, toxicity | identity_attack, insult, obscene, severe_toxicity, toxicity |

As the recall is better than precision for the trained model, the model mostly predicts more labels than there should be according to the gold labels. Sometimes though the model does not predict any label although there should be at least one. We have to note that the original Jigsaw data is not without flaws in itself and there might be mistakes in the labeling which might affect the model.