messages_ner = [{"role": "system", "content": """You are a language model that perform NER(named entity recognition) with the following text. Retrieve Only the list of named entities. Make sure that entity in whole must be only retireved once. Also make sure to merge the same entities with other name."""},
    {"role": "user", "content": """[Topic]
Darlene Remembers Duke, Jonathan Plays Fats (album)
[Text]
Darlene Remembers Duke, Jonathan Plays Fats is a 1982 album by Jo Stafford and Paul Weston in which they perform in character as Jonathan and Darlene Edwards. The duo put their own unique interpretation on the music of Duke Ellington and Fats Waller with Stafford singing deliberately off key, while Weston plays an out of tune piano. The album was issued by Corinthian Records (COR-117). "Billboard" reviewed the album when it was newly released, saying, "the sounds they achieve may well lead to another Grammy for the duo next year." Stafford and Weston, in their personas of Jonathan and Darlene Edwards, were interviewed by "Los Angeles Magazine" following the release of the album."""},
    {"role": "assistant", "content": """Darlene Remebers Duke, Jonathan Plays Fats (album)
Jo Stafford
Paul Weston
Jonathan Edwards
Darlene Edwards
Duke Ellington
Fats Wellor
Corinthian Records
COR-117
Billboard
Grammy
Los Angeles Magazine"""},
    {"role": "user", "content": """[Topic]
Felix R. de Zoysa
[Text]
Felix R. de Zoysa is the founding Chairman of Stafford Motor Company, Stafford International School and Atlas Hall in Sri Lanka. De Zoysa was the head of Auto & General Agencies who held the distributorship of Honda motor vehicles in Sri Lanka in 1960s. After the incorporation of both companies, Stafford Motor Company became the sole distributor for Honda motor vehicles. He was also nicknamed "Mr. Honda" of Sri Lanka."""},
    {"role": "assistant", "content": """Felix R. de Zoysa
Stafford Motor Company
Stafford International School
Atlas Hall
Sri Lanka
Auto & General Agencies
Honda motor vehicles
Mr. Honda
Sri Lanka
1960s"""},
    {"role": "user", "content": """[Topic]
Stafford Mills
[Text]
Stafford Mills is an historic textile mill complex located on County Street in Fall River, Massachusetts, USA. Founded in 1872, it is a well-preserved late-19th century textile complex, typical of the mills built in Fall River during its period of most rapid growth. It is noted in particular for its exceptionally fine Romanesque brick office building. The complex was added to the National Register of Historic Places in 1983."""},
    {"role": "assistant", "content": """Stafford Mills
County Street
Fall River
Massachusetts
USA
National Register of Historic Places
1872
1983"""},
]


messages_question_answering = [{"role": "system", "content":"""You are a language model that answers the given question using the given knowledge. The answer must be a single term that can be the answer of the question, without auxiliary terms."""},
{"role":"user","content":"""[Knowledge]
Hertfordshire is the county immediately north of London and is part of the East of England region, a mainly statistical unit. A significant minority of the population across all districts are City of London commuters. To the east is Essex, to the west is Buckinghamshire and to the north are Bedfordshire and Cambridgeshire.
[Question]
Where is the county of Hertfordshire located?
"""},
{"role":"assistant","content":"""England"""},
# {"role":"user","content":"""[Knowledge]
# The journal was established in December 1984 by founding editor-in-chief William J. Whelan under the auspices of the International Union of Biochemistry and Molecular Biology. Adam S. Wilkins became editor in January 1990. Originally published by ICSU Press and The Company of Biologists, "BioEssays" has been published by John Wiley & Sons since January 1998. Andrew Moore became editor-in-chief in August 2008.
# [Question]
# Who is the employer of John J. Collins"""},
# {"role":"assistant","content":"""None"""},
{"role":"user","content":"""[Knowledge]
The Game is a 1997 American mystery thriller film directed by David Fincher, starring Michael Douglas and Sean Penn, and produced by Propaganda Films and PolyGram Filmed Entertainment. It tells the story of a wealthy investment banker who is given a mysterious gift: participation in a game that integrates in strange ways with his everyday life. As the lines between the banker's real life and the game become more uncertain, hints of a large conspiracy become apparent.
[Question]
What studio distributed The Game?
"""},
{"role":"assistant","content":"""PolyGram Filmed Entertainment"""},]

messages_query_decomposition = [{"role": "system", "content": """You are a language model that divides the given query into list of subqueries and answers. Answers must be in form of $1,$2,$3, ... and the final answer must be $ANSWER."""},
    {"role": "user", "content": """"Where is the school that employs Charles Stafford located?"""},
    {"role": "assistant", "content": """<Q> Which school employs Charles Stafford?
<A> $1

<Q> Where is $1 located?
<A> $ANSWER"""},
    {"role": "user", "content":"""What year saw the creation of the region where the county of Hertfordshire is located?"""},
    {"role": "assistant", "content": """<Q> Where is Hertfordshire located?
<A> $1

<Q> In what region is $1 located? 
<A> $2

<Q> In what year was $2 created?
<A> $ANSWER"""},
    {"role": "user", "content":"""Are attendance remittances that important to the league whose trophy was won by an undefeated Arsenal team in the 2003-2004 season?"""},
    {"role": "assistant", "content": """
<Q> Which league's trophy was won by an undefeated Arsenal team in the 2003-2004 season?
<A> $1

<Q> Are attendance remittances important to $1?
<A> $ANSWER"""}, 
    {"role": "user", "content":"""What is something that is not consistent between the Jewish population in the country know for its Muslims and the people who traded slaves into the Middle East?"""},
    {"role": "assistant", "content": """<Q> What country is known for its Muslim population? <A> $1

<Q> What is the Jewish population in $1? <A> $2

<Q> Who were the people who traded slaves into the Middle East? <A> $3

<Q> What is something that is not consistent between the Jewish population in $1 and $3? <A> $ANSWER"""}, 
    {"role": "user", "content":"""Maria Leopoldina of the country where the Botanical Garden of where Alfred Biasiadeck was educated is the child of?"""},
    {"role": "assistant", "content": """<Q> Where was Alfred Biasiadeck educated? <A> $1

<Q> Where is the Botanical Garden of $1 located? <A> $2

<Q> Which country is $2 located in? <A> $3

<Q> Who is Maria Leopoldina of $3 the child of? <A> $ANSWER"""}, 
]