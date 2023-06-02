# Toxicity-classifier

This repository includes code for classifying toxic data as multi-label classification (6 labels + no label), multiclass/binary classification (toxic or not) and binary evaluation based on the multi-label classification. The goal is to make a decent classifier for Finnish using translated data. The data used is available in [huggingface](https://huggingface.co/datasets/TurkuNLP/wikipedia-toxicity-data-fi) and a model trained on this data is available [here](https://huggingface.co/TurkuNLP/bert-large-finnish-cased-toxicity). 

New annotated data for Finnish based on a sample from Suomi24 is available in the folder `annotations` in the file `all_annotations.tsv` and in [huggingface](https://huggingface.co/datasets/TurkuNLP/Suomi24-toxicity-annotated). The script for evaluating the annotated data is in the folder `predictions`.
Raw annotations are also included in the folder `annotations` in the subfolder `raw_annotations`.

## Demo
A demo can be found [here](https://github.com/TurkuNLP/toxicity-classifier/blob/main/prediction_scripts/Toxicity_Demo.ipynb)

## Presentations
A powerpoint file of the presentation held at NoDaLiDa 2023 can be found in the repository root.

## Citing
if you want to cite our work, use the bibtex below:

```
@inproceedings{eskelinen-etal-2023-toxicity,
    title = "Toxicity Detection in {F}innish Using Machine Translation",
    author = "Eskelinen, Anni  and
      Silvala, Laura  and
      Ginter, Filip  and
      Pyysalo, Sampo  and
      Laippala, Veronika",
    booktitle = "Proceedings of the 24th Nordic Conference on Computational Linguistics (NoDaLiDa)",
    month = may,
    year = "2023",
    address = "T{\'o}rshavn, Faroe Islands",
    publisher = "University of Tartu Library",
    url = "https://aclanthology.org/2023.nodalida-1.68",
    pages = "685--697",
    abstract = "Due to the popularity of social media platforms and the sheer amount of user-generated content online, the automatic detection of toxic language has become crucial in the creation of a friendly and safe digital space. Previous work has been mostly focusing on English leaving many lower-resource languages behind. In this paper, we present novel resources for toxicity detection in Finnish by introducing two new datasets, a machine translated toxicity dataset for Finnish based on the widely used English Jigsaw dataset and a smaller test set of Suomi24 discussion forum comments originally written in Finnish and manually annotated following the definitions of the labels that were used to annotate the Jigsaw dataset. We show that machine translating the training data to Finnish provides better toxicity detection results than using the original English training data and zero-shot cross-lingual transfer with XLM-R, even with our newly annotated dataset from Suomi24.",
}
```


## DISCLAIMER: THERE ARE EXPLICIT EXAMPLES FROM THE TEST DATA BELOW



## Examples of errors

| ID | Text | Gold label | Predicted label |
| -- | ---- | ---------- | --------------- |
| 9ddfac62e6bd6e09 | Älä valehtele.  Olet tyhmä. slavicmanifestdestiny194 | insult, obscene, toxicity | insult, obscene, toxicity |
| e62a2528247f06d3 | Mitä helvettiä kritiikkiä kaikesta paskasta, josta voisit syyttää häntä, tämä on parasta, mitä voit keksiä? | obscene, toxicity | obscene, toxicity |
| a069aa3a2019c4a0 | Brundage oli rasistinen sika mätänee helvetissä natsien kanssa - | identity_attack, insult, obscene, toxicity | identity_attack, insult, obscene, toxicity | 
| 9e175930f5d94034 | leikkaan sinut ja tapan sinut!!! HANKI ELÄMÄ, VITUN LUUSERI!!!! \n  leikkaan sinut ja tapan sinut!!! HANKI ELÄMÄ, VITUN LUUSERI!!!! \n  leikkaan sinut ja tapan sinut!!! HANKI ELÄMÄ, VITUN LUUSERI!!!! *--//-- (this continues for a while)* | insult, obscene, severe_toxicity, threat, label_toxicity | insult, obscene, severe_toxicity, threat, label_toxicity |
| 4312dfcc5da8adb9 | Obama on neekeri, eikä hänen pitäisi olla presidentti. | identity_attack, insult, toxicity | identity_attack, insult, toxicity |
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
