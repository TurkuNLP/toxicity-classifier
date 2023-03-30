## Label definitions taken from Perspective API

THREAT: Describes an intention to inflict pain, injury, or violence against an individual or group.
THREATENING: Language that is threatening or encouraging violence or harm, including self-harm.

PROFANITY: Swear words, curse words, or other obscene or profane language.

INSULT: Insulting, inflammatory, or negative comment towards a person or a group of people. Such comments are not necessarily identity specific.

IDENTITY ATTACK: Negative or hateful comments targeting someone because of their identity.

TOXICITY: A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.

SEVERE TOXICITY: A very hateful, aggressive, disrespectful comment or otherwise very likely to make a user leave a discussion or give up on sharing their perspective. This attribute is much less sensitive to more mild forms of toxicity, such as comments that include positive uses of curse words.

## Guidelines used for annotation:

### Obscene

swearwords, including mild expletives and misspelled, masked, or other variations
sexually explicit words/terminology that are not topically or contextually appropriate

### Threat

suicidal or self-harm comments, incitement to violence or self-harm, hypothetical situations and wishing harm to somebody
comments that are very unlikely to happen if not marked clearly as sarcasm
only threats towards people are annotated as threat

threats made by somebody else other than the writer NOT included
counterfactuals statements <!--- as in "if I was there I would have..." --->


### Insult

terms that are insulting towards groups of people (also in identity attack)
expletives with political group words such as "vitun demari/suvakki/persu"

negative insulting comments towards oneself, things other than people and hypothetical situations NOT included

PROBLEM: use of racist or rapist if true, target not clear

### Identity attack

comments that have no negative language but are still clearly negative

negative statements towards political groups or groups that nobody self-identifies with are NOT included (unless an insult)

### Toxicity

unreasonably expressed negative comments regardless of the target present and whether the target is known or not
mild or humoristic swearwords are NOT included
positive or neutral sexually explicit comments are NOT included

### Severe toxicity

comments that include only sexually explicit content
only one severely toxic element is needed to have this label and a comment is severely toxic even if the comment contains substantive content
target does not need to be present nor does the target matter


---

The final sample includes texts where the inter-annotator agreement was 1.0 or texts which we were able to resolve according to our discussion and the guidelines that followed.

---

## Inter-annotator agreement:

| Label | Initial (unanimous) | After discussion (unanimous) | Initial (at least 2/3) | After discussion (at least 2/3) |
|------ | ------------------- | ---------------------------- | ---------------------- | ------------------------------- |
| identity attack | 54,5 %  | 66,6 %  | 92 %  |  93,6 % |
| insult | 47,5 %   |  49,6 % |  94,5 % | 95,6 %  |
| severe toxicity | 63 %   | 66 %  |  92 % |   96,6 %   |
| threat |  82 %   | 80,3 %   |  98 % |  97,3 %   |
| toxicity | 58 %   | 54 %   |  64 %  |  89,6 %  |
| obscene | 69 %   | 62 %  |  97 %  | 96 % |



<!---
### Initial agreement

200 comments:

identity attack

     16 0.3333333333333333
     75 0.6666666666666666
    109 1.0

    unanimous: 54,5 %
    atleast 2/3: 92 %


insult
    
    11 0.3333333333333333
     94 0.6666666666666666
     95 1.0

    unanimous: 47,5 %
    atleast 2/3: 94,5 %


severe toxicity

     16 0.3333333333333333
     58 0.6666666666666666
    126 1.0

    unanimous: 63 %
    atleast 2/3: 92 %


threat

    4 0.3333333333333333
     32 0.6666666666666666
    164 1.0


    unanimous: 82 %
    atleast 2/3: 98 %

toxicity

    14 0.3333333333333333
     70 0.6666666666666666
    116 1.0

    unanimous: 58 %
    atleast 2/3: 64 %


100 comments:

obscene

    3 0.3333333333333333
     28 0.6666666666666666
     69 1.0

    unanimous: 69 %
    atleast 2/3: 97 %


### Agreement after discussion

300 comments:

identity attack

    19 0.3333333333333333
     81 0.6666666666666666
    200 1.0

    unanimous: 66,6 %
    atleast 2/3: 93,6 %

insult

     13 0.3333333333333333
    138 0.6666666666666666
    149 1.0
    
    unanimous: 49,6 %
    atleast 2/3: 95,6 %

severe toxicity

    10 0.3333333333333333
     92 0.6666666666666666
    198 1.0
 
    unanimous: 66 %
    atleast 2/3: 96,6 %

threat

      8 0.3333333333333333
     51 0.6666666666666666
    241 1.0

    unanimous: 80,3 %
    atleast 2/3: 97,3 %

toxicity

     31 0.3333333333333333
    107 0.6666666666666666
    162 1.0

    unanimous: 54 %
    atleast 2/3: 89,6 %

400 comments:

obscene

     16 0.3333333333333333
    136 0.6666666666666666
    248 1.0
    
    unanimous: 62 %
    atleast 2/3: 96 %

--->