import os.path
from dotenv import load_dotenv
from llama_index.llms.openai import OpenAI
from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    load_index_from_storage,
    Settings,
)
from llama_index.core.extractors import (
    SummaryExtractor,
    TitleExtractor
)
# from llama_index.llms.openai.utils import create_retry_decorator
# .env is not loading?
load_dotenv()

# llm_retry_decorator = create_retry_decorator(
#     max_retries=6,
#     random_exponential=True,
#     stop_after_delay_seconds=60,
#     min_seconds=1,
#     max_seconds=20,
# )

# OpenAI(temperature=0, model="gpt-3.5-turbo-0125")

# Set up extractors
title_extractor = TitleExtractor(nodes =5)
# summary_extractor = SummaryExtractor(summaries=["prev", "self", "next"])

# check if storage already exists

PERSIST_DIR = "./storage"
if not os.path.exists(PERSIST_DIR):
    # load the documents and create the index
    documents = SimpleDirectoryReader("data").load_data()
    index = VectorStoreIndex.from_documents(
        documents, transformations=[title_extractor]
    )
    # store it for later
    index.storage_context.persist(persist_dir=PERSIST_DIR)
else:
    # load the existing index
    storage_context = StorageContext.from_defaults(persist_dir=PERSIST_DIR)
    index = load_index_from_storage(storage_context)

# Set the model we are using, to gpt-4
Settings.llm = OpenAI(temperature=0.2, model="gpt-4-0125-preview")

# Either way we can now query the index
query_engine = index.as_query_engine()
response = query_engine.query(

"""I am a sociology student writing a essay about the AI hype and its relations to other scientific discipines and society as a whole.
The seminar I am writing the essay for is called Sociology of Science and Technology and the essay is due in 2 weeks.
I will provide you with my inital draft of the essay, that i need you to improve and expand upon.
I will provide you with my ideal structure for the essay formated as markdown, please respect the markdown formatting.
I will provide you with a .bib file for articles you may cite and you may add references to only those articles if they make sense necessary. 
References should look like this: 'Fox (@foxIrresponsibleResearchInnovation2018) mentions that ...' or 'As we see in @eynonMethodologyLegendRhetoric2021 ...' 
Additionally I will insert <TODO: ...> tags where I will give specific instructions for certain sections. THESE INSTRUCTIONS ARE MANDATORY.
The final essay should be around 3000 to 4000 words long, academic in nature but also multidisciplinary as the topic is complex and sociology, is for me a very broad research discipline. 
We will iterate over sections of the essay, heading by heading, and I will give you feedback on each section."""

"""# Analyzing AI Hype: Past Booms and the Present"
## Introduction
Topics: What is AI, how did it come to be?, What are hype cycles in technology?, Promise that they are related, and hook. 

## What is the state of the art of hype cycles in technological research?
Topics: Examples from other disciplines, Theory of Hype, Consequences of Hype, Hype applied to the current state of AI

## How has sociology engaged with AI during its past heydays?
Topics: Show old article, explain its main conclusions, Show that current considerations of AI are just the same, that there is not a lot of progress

## Does the current state of society and the media resemble a hype period?
Topics: Evaluate, show empirical data about current reporting. Add nuances, etc. 

## Is that necessarily an indicator for AI not leading to lasting societal changes? (Conclusion no)
Topics: Independent on if AI is hyped or not, it's impact on the job market and societal perception's of job value are real, The state of considering AI a cognizant being is just as relevant, this does not mean that it has diminished in importance"""

"""Here is the .bib file:
@article{foxIrresponsibleResearchInnovation2018,
	title = {Irresponsible {Research} and {Innovation}? {Applying} {Findings} from {Neuroscience} to {Analysis} of {Unsustainable} {Hype} {Cycles}},
	volume = {10},
	shorttitle = {Irresponsible {Research} and {Innovation}?},
	doi = {10.3390/su10103472},
	abstract = {The introduction of technological innovations is often associated with suboptimal decisions and actions during cycles of inflated expectations, disappointment, and unintended negative consequences. For brevity, these can be referred to as hype cycles. Hitherto, studies have reported hype cycles for many different technologies, and studies have proposed different methods for improving the introduction of technological innovations. Yet hype cycles persist, despite suboptimal outcomes being widely reported and despite methods being available to improve outcomes. In this communication paper, findings from exploratory research are reported, which introduce new directions for addressing hype cycles. Through reference to neuroscience studies, it is explained that the behavior of some adults in hype cycles can be analogous to that of irresponsible behavior among adolescents. In particular, there is heightened responsiveness to peer presence and potential rewards. Accordingly, it is argued that methods applied successfully to reduce irresponsible behavior among adolescents are relevant to addressing hype cycles, and to facilitating more responsible research and innovation. The unsustainability of hype cycles is considered in relation to hype about artificial intelligence (AI). In particular, the potential for human-beneficial AI to have the unintended negative consequence of being fatally unbeneficial to everything else in the geosphere other than human beings.},
	journal = {Sustainability},
	author = {Fox, Stephen},
	month = sep,
	year = {2018},
	pages = {3472},
	file = {Full Text:files/486/Fox - 2018 - Irresponsible Research and Innovation Applying Fi.pdf:application/pdf},
}
@article{eynonMethodologyLegendRhetoric2021,
	title = {Methodology, {Legend}, and {Rhetoric}: {The} {Constructions} of {AI} by {Academia}, {Industry}, and {Policy} {Groups} for {Lifelong} {Learning}},
	volume = {46},
	issn = {0162-2439},
	shorttitle = {Methodology, {Legend}, and {Rhetoric}},
	url = {https://doi.org/10.1177/0162243920906475},
	doi = {10.1177/0162243920906475},
	abstract = {Artificial intelligence (AI) is again attracting significant attention across all areas of social life. One important sphere of focus is education; many policy makers across the globe view lifelong learning as an essential means to prepare society for an ?AI future? and look to AI as a way to ?deliver? learning opportunities to meet these needs. AI is a complex social, cultural, and material artifact that is understood and constructed by different stakeholders in varied ways, and these differences have significant social and educational implications that need to be explored. Through analysis of thirty-four in-depth interviews with stakeholders from academia, commerce, and policy, alongside document analysis, we draw on the social construction of technology (SCOT) to illuminate the diverse understandings, perceptions of, and practices around AI. We find three different technological frames emerging from the three social groups and argue that commercial sector practices wield most power. We propose that greater awareness of the differing technical frames, more interactions among a wider set of relevant social groups, and a stronger focus on the kinds of educational outcomes society seeks are needed in order to design AI for learning in ways that facilitate a democratic education for all.},
	number = {1},
	urldate = {2024-03-14},
	journal = {Science, Technology, \& Human Values},
	author = {Eynon, Rebecca and Young, Erin},
	month = jan,
	year = {2021},
	note = {Publisher: SAGE Publications Inc},
	pages = {166--191},
	file = {Full Text PDF:files/488/Eynon and Young - 2021 - Methodology, Legend, and Rhetoric The Constructio.pdf:application/pdf},
}
@book{russellArtificialIntelligenceModern2016,
	address = {Boston Columbus Indianapolis New York San Francisco Upper Saddle River Amsterdam Cape Town Dubai London Madrid Milan Munich Paris Montreal Toronto Delhi Mexico City Sao Paulo Sydney Hong Kong Seoul Singapore Taipei Tokyo},
	edition = {Third edition, Global edition},
	series = {Prentice {Hall} series in artificial intelligence},
	title = {Artificial intelligence: a modern approach},
	isbn = {978-0-13-604259-4 978-1-292-15396-4},
	shorttitle = {Artificial intelligence},
	language = {en},
	publisher = {Pearson},
	author = {Russell, Stuart J. and Norvig, Peter},
	collaborator = {Davis, Ernest and Edwards, Douglas},
	year = {2016},
	file = {Russell and Norvig - 2016 - Artificial intelligence a modern approach.pdf:files/489/Russell and Norvig - 2016 - Artificial intelligence a modern approach.pdf:application/pdf},
}
@article{annetteruefWhatHappensHype2010,
	title = {What happens after a hype? {How} changing expectations affected innovation activities in the case of stationary fuel cells},
	volume = {22},
	doi = {10.1080/09537321003647354},
	abstract = {Innovation processes are influenced by the dynamics of expectations. This paper addresses the question of what happens after a hype. It takes a closer look at the case of stationary fuel cells, for which a hype could be identified in 2001 followed by a clear downscaling of expectations and disappointment. Innovation activities, however, remained largely unaffected by the disappointment. We offer two explanations. First, only generalised expectations were adjusted after the hype, while overarching expectations (frames) remained stable and continued to legitimate the technology. Second, emerging institutional structures lead to increasing positive externalities thus stabilising ongoing innovation activities. These institutionalisation processes, again, were supported by a transformation of promises into requirements during the hype and the fact that the frames continued to legitimise the technology afterwards. We conclude with the proposition to differentiate disappointments according to which type of expec...},
	number = {3},
	journal = {Technology Analysis \& Strategic Management},
	author = {{Annette Ruef} and Ruef, Annette and {Jochen Markard} and Markard, Jochen},
	month = mar,
	year = {2010},
	doi = {10.1080/09537321003647354},
	note = {MAG ID: 2089110760},
	pages = {317--338},
}
@article{benwilliamsonHistoricalThreadsMissing2020,
	title = {Historical threads, missing links, and future directions in {AI} in education},
	volume = {45},
	doi = {10.1080/17439884.2020.1798995},
	abstract = {Artificial intelligence has become a routine presence in everyday life. Accessing information over the Web, consuming news and entertainment, the performance of financial markets, the ways surveill...},
	number = {3},
	journal = {Learning, Media and Technology},
	author = {{Ben Williamson} and Williamson, Ben and {Rebecca Eynon} and Eynon, Rebecca},
	month = jul,
	year = {2020},
	doi = {10.1080/17439884.2020.1798995},
	note = {MAG ID: 3046530224},
	pages = {223--235},
	file = {Accepted Version:files/495/Ben Williamson et al. - 2020 - Historical threads, missing links, and future dire.pdf:application/pdf},
}
@article{ruthlevitasUtopiaMethod2013,
	title = {Utopia as {Method}},
	doi = {10.1057/9781137314253},
	abstract = {Utopia should be understood as a method rather than a goal. This book rehabilitates utopia as a repressed dimension of the sociological and in the process produces the Imaginary Reconstitution of Soci},
	author = {{Ruth Levitas} and Levitas, Ruth},
	month = jul,
	year = {2013},
	doi = {10.1057/9781137314253},
	note = {MAG ID: 1565146988},
}
@article{woolgarWhyNotSociology1985a,
	title = {Why {Not} a {Sociology} of {Machines}? {The} {Case} of {Sociology} and {Artificial} {Intelligence}},
	volume = {19},
	issn = {0038-0385},
	shorttitle = {Why {Not} a {Sociology} of {Machines}?},
	url = {https://www.jstor.org/stable/42853468},
	abstract = {In the light of the recent growth of artificial intelligence (AI), and of its implications for understanding human behaviour, this paper evaluates the prospects for an association between sociology and artificial intelligence. Current presumptions about the distinction between human behaviour and artificial intelligence are identified through a survey of discussions about AI and 'expert systems'. These discussions exhibit a restricted view of sociological competence, a marked rhetoric of progress and a wide variation in assessments of the state of the art. By drawing upon recent themes in the social study of science, these discussions are shown to depend on certain key dichotomies and on an interpretive flexibility associated with the notions of intelligence and expertise. The range of possible associations between sociology and AI reflects the extent to which we are willing to adopt these features of AI discourse. It is suggested that one of the more important options is to view the AI phenomenon as an occasion for reassessing the central axiom of sociology that there is something distinctively 'social' about human behaviour.},
	number = {4},
	urldate = {2024-03-17},
	journal = {Sociology},
	author = {Woolgar, Steve},
	year = {1985},
	note = {Publisher: Sage Publications, Ltd.},
	pages = {557--572},
	file = {JSTOR Full Text PDF:files/502/Woolgar - 1985 - Why Not a Sociology of Machines The Case of Socio.pdf:application/pdf},
}
@article{liuSociologicalPerspectivesArtificial2021a,
	title = {Sociological perspectives on artificial intelligence: {A} typological reading},
	volume = {15},
	copyright = {© 2021 The Authors. Sociology Compass published by John Wiley \& Sons Ltd.},
	issn = {1751-9020},
	shorttitle = {Sociological perspectives on artificial intelligence},
	url = {https://onlinelibrary.wiley.com/doi/abs/10.1111/soc4.12851},
	doi = {10.1111/soc4.12851},
	abstract = {Interest in applying sociological tools to analysing the social nature, antecedents and consequences of artificial intelligence (AI) has been rekindled in recent years. However, for researchers new to this field of enquiry, navigating the expansive literature can be challenging. This paper presents a practical way to help these researchers to think about, search and read the literature more effectively. It divides the literature into three categories. Research in each category is informed by one analytic perspective and analyses one “type” of AI. Research informed by the “scientific AI” perspective analyses “AI” as a science or scientific research field. Research underlain by the “technical AI” perspective studies “AI” as a meta-technology and analyses its various applications and subtechnologies. Research informed by the “cultural AI” perspective views AI development as a social phenomenon and examines its interactions with the wider social, cultural, economic and political conditions in which it develops and by which it is shaped. These analytic perspectives reflect the evolution of “AI” from chiefly a scientific research subject during the twentieth century to a widely commercialised innovation in recent decades and increasingly to a distinctive socio-cultural phenomenon today.},
	language = {en},
	number = {3},
	urldate = {2024-03-17},
	journal = {Sociology Compass},
	author = {Liu, Zheng},
	year = {2021},
	note = {\_eprint: https://onlinelibrary.wiley.com/doi/pdf/10.1111/soc4.12851},
	keywords = {artificial intelligence, digital technology, sociology, typology},
	pages = {e12851},
	file = {Full Text PDF:files/504/Liu - 2021 - Sociological perspectives on artificial intelligen.pdf:application/pdf;Snapshot:files/505/soc4.html:text/html},
}
@article{joyceSociologyArtificialIntelligence2021,
	title = {Toward a {Sociology} of {Artificial} {Intelligence}: {A} {Call} for {Research} on {Inequalities} and {Structural} {Change}},
	volume = {7},
	issn = {2378-0231},
	shorttitle = {Toward a {Sociology} of {Artificial} {Intelligence}},
	url = {https://doi.org/10.1177/2378023121999581},
	doi = {10.1177/2378023121999581},
	abstract = {This article outlines a research agenda for a sociology of artificial intelligence (AI). The authors review two areas in which sociological theories and methods have made significant contributions to the study of inequalities and AI: (1) the politics of algorithms, data, and code and (2) the social shaping of AI in practice. The authors contrast sociological approaches that emphasize intersectional inequalities and social structure with other disciplines’ approaches to the social dimensions of AI, which often have a thin understanding of the social and emphasize individual-level interventions. This scoping article invites sociologists to use the discipline’s theoretical and methodological tools to analyze when and how inequalities are made more durable by AI systems. Sociologists have an ability to identify how inequalities are embedded in all aspects of society and to point toward avenues for structural social change. Therefore, sociologists should play a leading role in the imagining and shaping of AI futures.},
	language = {en},
	urldate = {2024-03-17},
	journal = {Socius},
	author = {Joyce, Kelly and Smith-Doerr, Laurel and Alegria, Sharla and Bell, Susan and Cruz, Taylor and Hoffman, Steve G. and Noble, Safiya Umoja and Shestakofsky, Benjamin},
	month = jan,
	year = {2021},
	note = {Publisher: SAGE Publications},
	pages = {2378023121999581},
	file = {SAGE PDF Full Text:files/507/Joyce et al. - 2021 - Toward a Sociology of Artificial Intelligence A C.pdf:application/pdf},
}
@article{simonenataleImaginingThinkingMachine2020,
	title = {Imagining the thinking machine: technological myths and the rise of {Artificial} {Intelligence}},
	volume = {26},
	doi = {10.1177/1354856517715164},
	abstract = {This article discusses the role of technological myths in the development of artificial intelligence (AI) technologies from 1950s to the early 1970s. It shows how the rise of AI was accompanied by ...},
	number = {1},
	journal = {Convergence},
	author = {{Simone Natale} and Natale, Simone and {Andrea Ballatore} and Ballatore, Andrea},
	month = feb,
	year = {2020},
	doi = {10.1177/1354856517715164},
	note = {MAG ID: 2623152824},
	pages = {3--18},
	file = {Full Text:files/513/Simone Natale et al. - 2020 - Imagining the thinking machine technological myth.pdf:application/pdf},
}
@article{madsborupSociologyExpectationsScience2006,
	title = {The sociology of expectations in science and technology},
	volume = {18},
	doi = {10.1080/09537320600777002},
	abstract = {Technology Scenarios Research Programme, Systems Analysis Department, Riso National Laboratory, Roskilde, Denmark, Science and Technology Unit, Department of Sociology, University of York, UK, Cirus-Innovation Research in Utility Sectors Eawag-Swiss Federal Institute for Aquatic Science and Technology, Dubendorf, Switzerland, Department of Innovation Studies, Copernicus Institute for Sustainable Development and Innovation, University of Utrecht, The Netherlands},
	journal = {Technology Analysis \& Strategic Management},
	author = {{Mads Borup} and Borup, Mads and {Nik Brown} and Brown, Nik and {Kornelia Konrad} and Konrad, Kornelia and {Harro van Lente} and van Lente, Harro},
	month = aug,
	year = {2006},
	doi = {10.1080/09537320600777002},
	note = {MAG ID: 2049941951},
	pages = {285--298},
}
@article{jackstilgoeMachineLearningSocial2018,
	title = {Machine learning, social learning and the governance of self-driving cars:},
	volume = {48},
	doi = {10.1177/0306312717741687},
	abstract = {Self-driving cars, a quintessentially ‘smart’ technology, are not born smart. The algorithms that control their movements are learning as the technology emerges. Self-driving cars represent a high-stakes test of the powers of machine learning, as well as a test case for social learning in technology governance. Society is learning about the technology while the technology learns about society. Understanding and governing the politics of this technology means asking ‘Who is learning, what are they learning and how are they learning?’ Focusing on the successes and failures of social learning around the much-publicized crash of a Tesla Model S in 2016, I argue that trajectories and rhetorics of machine learning in transport pose a substantial governance challenge. ‘Self-driving’ or ‘autonomous’ cars are misnamed. As with other technologies, they are shaped by assumptions about social needs, solvable problems, and economic opportunities. Governing these technologies in the public interest means improving soci...},
	number = {1},
	journal = {Social Studies of Science},
	author = {{Jack Stilgoe} and Stilgoe, Jack},
	month = feb,
	year = {2018},
	doi = {10.1177/0306312717741687},
	pmid = {29160165},
	note = {MAG ID: 2607757716},
	pages = {25--56},
	file = {Submitted Version:files/514/Jack Stilgoe and Stilgoe - 2018 - Machine learning, social learning and the governan.pdf:application/pdf},
}
@article{harrovanlenteComparingTechnologicalHype2013,
	title = {Comparing technological hype cycles: {Towards} a theory},
	volume = {80},
	doi = {10.1016/j.techfore.2012.12.004},
	abstract = {The notion of ‘hype’ is widely used and represents a tempting way to characterize developments in technological fields. The term appears in business as well as in academic domains. Consultancy firms offer technological hype cycle models to determine the state of development of technological fields in order to facilitate strategic investment decisions. In Science, Technology and Innovation Studies the concept of hype is considered in studies on the dynamics of expectations in innovation processes, which focuses on the performative force of expectations. What is still lacking is a theory of hype patterns that is able to explain the different shapes of hype cycles in different contexts. In this paper we take a first step towards closing this gap by studying and comparing the results of case studies on three hypes in three different empirical domains: voice over internet protocol (VoIP), gene therapy and high-temperature superconductivity. The cases differ in terms of the type of technology and the characteristics of the application environment. We conclude that hype patterns indeed vary a lot, and that the interplay of expectations at different levels affects the ability of a field to cope with hype and disappointment.},
	number = {8},
	journal = {Technological Forecasting and Social Change},
	author = {{Harro van Lente} and van Lente, Harro and {Charlotte Spitters} and Spitters, Charlotte and {Alexander Peine} and Peine, Alexander},
	month = oct,
	year = {2013},
	doi = {10.1016/j.techfore.2012.12.004},
	note = {MAG ID: 1976529177},
	pages = {1615--1628},
}"""

"""# Analyzing AI Hype: Past Booms and the Present

The exploration of artificial intelligence (AI) and its cyclical journey through periods of intense enthusiasm and subsequent disillusionment provides a fascinating lens through which to view the evolution of technology and its societal implications. The inception of AI can be traced back to the pioneering work of Warren McCulloch and Walter Pitts in 1943, who proposed a model of artificial neurons based on a combination of knowledge from neuroscience, formal logic, and computation theory. This model laid the groundwork for the first neural network computer, constructed by Marvin Minsky and Dean Edmonds at Harvard in 1950, marking a seminal moment in the development of AI (@Russell & Norvig, 1995). 
The early achievements of AI, though modest by today's standards, were groundbreaking for their time. Computers, previously regarded as mere calculators, began performing tasks that bore a resemblance to human cognition, challenging prevailing assumptions about the capabilities of machines. This era, often referred to by John McCarthy as the “Look, Ma, no hands!” period, was characterized by a series of demonstrations where AI systems performed tasks previously thought to be exclusive to human intelligence.
However, the optimism of AI researchers, as epitomized by Herbert Simon's bold predictions in 1957, often outpaced the actual progress of the technology. Simon's forecasts, including a computer becoming a chess champion and proving significant mathematical theorems within a decade, were not realized within his optimistic timeframe. Instead, they materialized over a span of 40 years, highlighting a pattern of overestimation that has recurred throughout AI's history. This tendency for overconfidence was largely due to the promising yet limited performance of early AI systems on simple tasks, which did not scale to more complex problems as anticipated (@Natale & Ballatore, 2017).
The phenomenon of AI hype is not unique to this field but is part of a broader pattern observed across various technological domains. Research into hype cycles reveals that many technologies, including biotechnology, self-driving vehicles, and hydrogen fuel cells, undergo similar trajectories of inflated expectations followed by periods of disillusionment (@Fox, 2018; @Stilgoe, 2017; @Ruef & Markard, 2010). These cycles are characterized by a surge in media attention and investment driven by high expectations, which eventually confront the reality of technological challenges and limitations.
The concept of 'hype' in technology encompasses the dynamics of expectations that shape the development and adoption of new innovations. High expectations can mobilize resources, attract stakeholders, and confer legitimacy on emerging technologies. However, when these expectations exceed the actual capabilities of the technology, they can lead to an 'overshoot' that undermines credibility and leads to disillusionment. This pattern is well-documented in both business and academic studies, with consultancy firms developing models such as the Gartner hype cycle to navigate these dynamics. Science, Technology, and Innovation Studies have further enriched our understanding by examining the performative nature of hype and its impact on innovation processes (@Lente et al., 2013).
In the context of AI, the current wave of enthusiasm must be critically examined against the backdrop of these historical cycles. While the potential of AI to transform industries, healthcare, and daily life is undeniable, the lessons from past hype cycles caution against unchecked optimism. The interplay between media narratives, public expectations, and technological development shapes the trajectory of AI, making it imperative to foster a balanced discourse that acknowledges both the potential and the limitations of AI technology.
By situating the current state of AI within this historical and sociological framework, we gain valuable insights into the forces that drive technological hype and the complex interplay between innovation, societal expectations, and media narratives. This perspective not only enriches our understanding of AI's development but also informs a more nuanced approach to navigating its future trajectory.

## What is the state of the art of hype cycles in technological research?
In the realm of technological innovation, the concept of hype cycles plays a crucial role in shaping the trajectory of emerging technologies. These cycles, characterized by fluctuating levels of public and investor interest, significantly impact the development, adoption, and eventual maturation of technologies. The theory of hype cycles, initially conceptualized by Gartner, a leading research and advisory company, provides a framework for understanding these fluctuations. According to Gartner, technologies typically pass through five phases: the "Technology Trigger," "Peak of Inflated Expectations," "Trough of Disillusionment," "Slope of Enlightenment," and finally, the "Plateau of Productivity" (@harrovanlenteComparingTechnologicalHype2013). This model helps stakeholders navigate the complex dynamics of technological evolution, offering insights into when to invest, when to develop, and when to implement new technologies.       
The consequences of hype are multifaceted. On one hand, heightened expectations can drive rapid advancements and innovation by attracting investment and talent to the field. On the other hand, unrealistic expectations can lead to disappointment and disillusionment, potentially stalling progress and deterring future investment. The study of hype cycles, therefore, is not merely an academic exercise but a practical tool for managing the lifecycle of technological innovations.
The current state of artificial intelligence (AI) serves as a prime example of a technology navigating through its hype cycle. AI has experienced several waves of heightened expectations followed by periods of disillusionment since its inception in the mid-20th century. Each cycle has been driven by breakthroughs in technology, such as the development of neural networks and deep learning algorithms, which have periodically reignited the public and investors' imaginations about the potential of AI. However, these periods of enthusiasm have often been followed by setbacks, as the challenges of implementing AI in practical, real-world applications become apparent (@jackstilgoeMachineLearningSocial2018).
The sociological perspective offers a unique lens through which to examine hype cycles, emphasizing the social dynamics and institutional practices that contribute to the rise and fall of technological expectations. Sociologists have highlighted how hype not only reflects but also shapes technological development, influencing which projects receive funding, which research directions are pursued, and how technologies are ultimately implemented in society. This perspective underscores the importance of critically examining the social processes that underlie technological hype, moving beyond individual technologies to consider the broader socio-technical systems in which they are embedded (@joyceSociologyArtificialIntelligence2021).
Moreover, the interplay between media narratives, public expectations, and technological development is critical in shaping the trajectory of AI and other emerging technologies. Media representations can amplify the perceived potential of technologies, contributing to the peak of inflated expectations, while also playing a role in the subsequent disillusionment as challenges and limitations become more apparent. Understanding this dynamic is essential for navigating the hype cycle effectively, ensuring that technologies can progress toward their plateau of productivity without being unduly hindered by unrealistic expectations or premature disillusionment (@simonenataleImaginingThinkingMachine2020). 

## How has sociology engaged with AI during its past heydays?
The intersection of sociology and artificial intelligence (AI) is not a novel area of inquiry. Historical engagement with AI by sociologists has been profound, offering critical insights into the sociotechnical dynamics of this emerging technology. Steve Woolgar's seminal work in the mid-1980s laid the groundwork for a sociological perspective on AI, challenging the prevailing dichotomies between human intelligence and machine capabilities. Woolgar argued for a sociology that goes beyond merely adopting the discourse of AI, advocating instead for an empirical investigation into the distinctions and relationships that characterize the human-machine interface. This involves scrutinizing the public pronouncements of AI proponents in relation to the day-to-day activities of AI researchers, thereby uncovering the social processes underpinning the development of AI (@Woolgar, 1985).
Woolgar's call for a sociology of machines proposed a radical reevaluation of the basic axioms of sociology, particularly the assumption that human behavior is distinctively 'social' and fundamentally different from machine activity. This challenge to sociological orthodoxy invites a reconsideration of our understanding of behavior, action, and agency, urging sociologists to question why the discipline has traditionally excluded machine-like activity from its purview. The advent of AI, with its attempts to replicate or simulate human intelligence, provides a unique empirical opportunity to probe the limits of the distinction between human behavior and machine activity. Woolgar's work thus sets the stage for a broader inquiry into the social dimensions of AI, encouraging sociologists to explore how societal conceptions of intelligence and machine activity shape our understanding of technology and its implications for society.      
Contemporary sociological research on AI mirrors Woolgar's early insights, extending the analysis to the complex interplay between AI technologies, societal expectations, and the media narratives that shape public perceptions of AI. Recent studies have emphasized the importance of examining the social shaping of AI in practice, highlighting how AI systems are embedded within broader sociotechnical systems that reflect and reproduce societal values, power dynamics, and inequalities (@Joyce et al., 2021). This body of work builds on Woolgar's foundational arguments, employing sociological theories and methods to analyze how AI technologies are developed, implemented, and understood within specific social, cultural, and institutional contexts.
In summary, the engagement of sociology with AI, from its early days to the present, reflects a continuous effort to unpack the sociotechnical entanglements of AI technologies. By drawing on Woolgar's pioneering work and its contemporary extensions, sociologists are well-positioned to contribute critical insights into the development and societal implications of AI, challenging simplistic narratives and highlighting the complex realities of AI as a sociotechnical phenomenon.

## Current Societal and Media Perspectives on AI
The contemporary societal and media landscape is saturated with discussions about artificial intelligence (AI), often painting a picture of a future radically transformed by AI technologies. This narrative is not without its merits, as  AI has indeed made significant strides in various fields, from healthcare diagnostics to autonomous vehicles. However, the intensity and nature of the current discourse around AI bear the hallmarks of a hype period, characterized by inflated expectations and speculative projections about the technology's potential impact on society.
A critical examination of media reports and public discourse reveals a pattern of sensationalism and optimism that frequently overshadows the nuanced realities of AI development and implementation. For instance, headlines often tout AI's capabilities in surpassing human performance in specific tasks, such as game playing or image recognition, without acknowledging the limitations of these systems in more complex, real-world scenarios. This portrayal contributes to a public perception of AI as an omnipotent force poised to revolutionize every aspect of human life, from work to social interactions (@jackstilgoeMachineLearningSocial2018).
However, a nuanced approach is necessary to distinguish between the possible and the not possible in AI. While AI technologies have indeed advanced significantly, their current capabilities are often more limited than media narratives suggest. For example, AI systems excel in tasks with clear rules and objectives but struggle with ambiguity, context, and tasks requiring common sense or ethical judgment. This discrepancy between the hype and the reality of AI underscores the importance of critically evaluating the claims made about AI's potential and the societal implications of its widespread adoption (@simonenataleImaginingThinkingMachine2020).
Moreover, the current hype around AI is not merely a matter of media sensationalism but is also influenced by the interests of various stakeholders, including tech companies, investors, and policymakers. These actors have a vested interest in promoting an optimistic view of AI, as it can attract investment, drive research and development, and shape policy agendas. However, this promotion often glosses over the challenges and risks associated with AI, including ethical concerns, potential job displacement, and the exacerbation of social inequalities (@joyceSociologyArtificialIntelligence2021).
In conclusion, while the current societal and media perspectives on AI reflect a period of hype, it is crucial to adopt a critical and nuanced approach to understanding AI's capabilities and limitations. By examining the evidence behind the claims made about AI and considering the interests driving the hype, we can develop a more balanced and informed view of AI's potential role in society. This approach not only helps temper unrealistic expectations but also highlights the areas where AI can genuinely contribute to addressing societal challenges, guiding the development and implementation of AI technologies in a responsible and equitable manner.

## AI's Impact Beyond Hype
<TODO: Medium changes allowed.>Regardless of whether AI is currently experiencing a hype cycle, its impact on the job market and societal perceptions of job value is undeniable. The consideration of AI as a cognizant being remains a relevant topic of discussion, underscoring the ongoing significance of AI in societal change.
<TODO: Medium changes allowed.>In conclusion, the history of AI is marked by cycles of hype and disillusionment, yet the sociological engagement with AI offers valuable insights into the ongoing discourse. The current enthusiasm for AI, while reflective of past patterns, does not diminish the technology's potential impact on society. As such, the exploration of AI's development and its societal implications remains a critical area of inquiry."""

"""Lets improve the last segment: ## AI's Impact Beyond Hype
This segment is quite short, but it should also be between 400 and 500 words long. Please search in the provided articles. This segment should be the conclusion of this article, 
showing that the current AI hype is not really new, and that while AI undoubtedly has large impacts on society, they may differ from what is currently being discussed. It will uderscore this opinion
with content discussed in above segments. 
Please consider the above points especially the <TODO: ...> tags and the provided articles."""
)

print(response)

