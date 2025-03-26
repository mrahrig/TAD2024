####################################
#install.packages("topicmodels")
#install.packages("devtools")
#install.packages("webshot2")
# install.packages("stopwords")
# install.packages("topicmodels")
# install.packages("ldatuning")
# install.packages("stm")
# install.packages('jsonlite')
# install.packages('ndjson')
# install.packages('rjson')
# install.packages("tm")
# install.packages('httr2')
# install.packages("tidyverse")
# install.packages('topicmodels')
# install.packages('ldatuning')
# install.packages("stopwords")
# install.packages("quanteda")
# install.packages('cluster')
# install.packages('stats')
# install.packages('factoextra') 
# install.packages('SnowballC') 
# install.packages('textclean')
# install.packages('lexicon') 
# install.packages("writexl")
#install.packages("klaR")
library(caret)
library(webshot2)
library(plotly)
library(klaR)
library(writexl)
library(tm)
library(SnowballC)
library(ldatuning)
library(topicmodels)
library(devtools)
library(stopwords)
library(quanteda)
library(stm)
library(jsonlite)
library(ndjson)
library(rjson)
library(tm)
library(httr2)
library(tidyverse)
library(cluster)
library(factoextra)
library(stats)
library(httr)
library(purrr)
library(dplyr)
library(tibble)
library(e1071)
library(textclean)
library(lexicon)
########################

setwd("/Users/meganrahrig/Desktop/JHU/Text as Data - SummerQ24/Final")

codyko_posts <- read_csv('codyko_subreddit_data.csv')
codyko_posts <- codyko_posts %>%
  distinct(id, .keep_all = TRUE)

codyko_posts <- codyko_posts %>%
  filter(!is.na(body) & body != "")

####################
#Stopwords
#####################
# Define stopword lists
tm_stopwords <- stopwords("en")  
snowball_stopwords <- wordStem(stopwords("en"))  
quanteda_stopwords <- stopwords("en")  
tidytext_stopwords <- tidytext::stop_words$word  
custom_exclusions <- c("i's", "ive", "im", "he", "hes", "just", "don't", "it‚Äôs", "i‚Äôm", "i‚Äôve", "he‚Äôs")
replacements <- rep("", length(custom_exclusions))

# Combine all stopwords and make them unique
expanded_stopwords <- c(
  stopwords::stopwords("en"),
  stopwords::stopwords(source = "stopwords-iso"),
  stopwords::stopwords(source = "smart"),
  stopwords::stopwords(source = "marimo"),
  stopwords::stopwords(source = "nltk"),
  tm_stopwords,
  snowball_stopwords,
  quanteda_stopwords,
  tidytext_stopwords,
  custom_exclusions,
 c('gt', 'us', 'get', 't', 'e', 'r', 'like', 'just')
) %>% unique()


##################################################################
#Combine Data
##################################################
combined_data <- tibble(
  post_id = codyko_posts$id,
  title = codyko_posts$title,
  body = codyko_posts$body,
  date = as.Date(codyko_posts$created_utc, format="%Y-%m-%d")
) %>%
  mutate(
    combined_text = paste(codyko_posts$title, codyko_posts$body, sep = " - ")
  )


# Replace problematic sequences with the correct ones
combined_data$combined_text <- gsub("‚Äôm", "'m", combined_data$combined_text)
combined_data$combined_text <- gsub("‚Äôs", "'s", combined_data$combined_text)
combined_data$combined_text <- gsub("‚Äôve", "'ve", combined_data$combined_text)
combined_data$combined_text <- gsub("‚Äôt", "'t", combined_data$combined_text)
combined_data$combined_text <- gsub("â\u0080\u009aÃ\u0084Ã´", "'", combined_data$combined_text)
combined_data$combined_text <- gsub("â", "'", combined_data$combined_text)
combined_data$combined_text <- gsub("ã", "a", combined_data$combined_text)
combined_data$combined_text <- gsub("[^\x20-\x7E]", "", combined_data$combined_text)
combined_data$combined_text <- gsub("[^\x20-\x7E]", " ", combined_data$combined_text)

print(head(combined_data$combined_text))

print(head(combined_data, 10))

##################################################################
#Corpus
##################################################################
codyko_corpus <- corpus(combined_data, text_field = "combined_text")

codyko_dfm <- codyko_corpus %>%
  tokens(remove_punct = TRUE, 
         remove_url = TRUE,
         remove_symbols = TRUE) %>%
  tokens_replace(custom_exclusions, replacements) %>%
  tokens_tolower() %>%
  tokens_remove(expanded_stopwords) %>%
  dfm()

#dfm
codyko_dfm <- dfm(tokens(codyko_corpus))

#
codyko_dfm <- codyko_corpus %>%
  tokens(remove_punct = TRUE, 
         remove_url = TRUE,
         remove_symbols = TRUE) %>%
  tokens_tolower() %>%  # Ensure tokens are lowercased
  tokens_remove(expanded_stopwords) %>%
  dfm() %>%
  dfm_trim(min_termfreq = 0.6,
           termfreq_type = 'quantile',
           max_docfreq = 0.4,
           docfreq_type="prop")

#Convert to DTM (which is just a DFM)
codyko_dtm <- convert(codyko_dfm, to='topicmodels')
##################################################################
#LDA
##################################################################
#Write a function to try several values for K and store the results for each K to review.
lda_k_func <- function(dtm, k_values, num_terms = 6) {
  models <- list()
  
  for (k in k_values) {
    lda_model <- LDA(dtm, k = k, method = "VEM")
    terms_k <- terms(lda_model, num_terms)
    models[[paste0("LDA_k_", k)]] <- list(model = lda_model, terms = terms_k)
    cat("\nTerms for LDA model with", k, "topics:\n")
    print(terms_k)
  }
  return(models)
}

#Range (could not make it work with 1-10)
k_values <- 2:5

#Fit models and store terms for each k
lda_k_models <- lda_k_func(codyko_dtm, k_values)


#######################################
#Sentiment
#####################################
# Load the Bing sentiment lexicon
library(tidytext)
bing_lexicon <- get_sentiments("bing")

tidy_dfm <- tidy(codyko_dfm)

tidy_sentiment <- tidy_dfm %>%
  inner_join(bing_lexicon, by = c("term" = "word"))

#Scores
sentiment_scores <- tidy_sentiment %>%
  group_by(document) %>%
  summarize(sentiment = sum(ifelse(sentiment == "positive", count, -count), na.rm = TRUE))
#Merge
sentiment_data <- combined_data %>%
  mutate(document = paste0("text", row_number())) %>%
  inner_join(sentiment_scores, by = "document")
#print
print(sentiment_data)
#plot sentiment over time
p <- ggplot(sentiment_data, aes(x = date, y = sentiment)) +
  geom_line(color = "blue") +
  geom_smooth(method = "loess", color = "red", se = FALSE) +  # Adding a smooth average line without confidence intervals
  labs(title = "Sentiment Over Time",
       x = "Timestamp",
       y = "Sentiment Score") +
  theme_minimal()

#Convert
p_interactive <- ggplotly(p)
p_interactive

htmlwidgets::saveWidget(p_interactive, "sentiment_plot.html")
ggsave("sentiment_plot.png", plot = p)

################
##################################################################
#Create Labels
##################################################################
# post_ids <- c("1e4l4mo",	"1e0aqb6",	"1e9oz6i",	"1e7lc9e",	"1e43b2h",	"1e614jl",	"17hmdeq",	"1e3d80s"
#               ,	"1e8klhz",	"1ejkkc0",	"1e44ror",	"1ec6zlr",	"1eb8cho",	"1eepcc5",	"1e6e7ep",	"1el14r9"
#               ,	"hfq3c6",	"1envfw4",	"1e8j4dr",	"14673o9",	"1ar4zit",	"1e555o3",	"1e69doe",	"1ecug9x"
#               ,	"1ea8aa5",	"1e7bafq",	"1ehu6bs",	"1edq1k2",	"1e5cze5",	"1e4z4p9",	"1e9r5iw",	"1e8xvzu"
#               ,	"18j6krz",	"1e7ywf2",	"10uzzq8",	"1e9h8tg",	"1e65db4",	"1e9i0o3",	"16tm6d3",	"1eonq7a"
#               ,	"1e4549q",	"17kga00",	"1e7gogm",	"1e5rx91",	"16dm1l6",	"1eckcm3",	"1emmhfb",	"1ee9afp"
#               ,	"1edwtyn",	"1dsz578",	"1e4qvay",	"1e8rm94",	"179llq2",	"12h6lbp",	"1efeo56",	"1ejx9ae"
#               ,	"zpzpp2",	"1e52jlt",	"1eii4w7",	"1e4ijjv",	"1egcie6",	"1eddte0",	"1elv3gh",	"13p3ejn"
#               ,	"131wman",	"1en9gtn",	"1e4w3so",	"1941781",	"1db7smd",	"1771vac",	"1e74iwf",	"1ecl6ns"
#               ,	"1el8n4y",	"1e9y2f6",	"o91zei",	"189t41y",	"1agr7qq",	"1e7v2zi",	"1adwvg8",	"1e6feut"
#               ,	"1ejbm0z",	"1ajrjvr",	"l0uoie",	"1e86g4u",	"17md4ht",	"1e4672s",	"1ef9q5i",	"149jdwy"
#               ,	"19bpall",	"1ae2psw",	"1eedq5u",	"16o4n37",	"peyzvh",	"13bp11s",	"1e3mcwf",	"1397w74"
#               ,	"1eh7i57",	"1334oqq",	"1e9ckpb",	"1ecw41s",	"1enqb1d",	"17og34a",	"1e75il7",	"tbev40"
#               ,	"1e43ndd",	"nw6qyy",	"195hrnw",	"1efslbp",	"1e4adli",	"ulas8w",	"1e570ai",	"1ed2qpr",	"1e6se4f"
#               ,	"1efgnsq",	"1e8t8i4",	"1edt76k",	"1eb32ut",	"1e5nr59",	"1e3zjev",	"12zz5t0",	"1bw4ty4"
#               ,	"1ef4biy",	"1e52zp0",	"1brnwgp",	"1amdml1",	"1ebboxf",	"n7ubfk",	"1afpvbl",	"14iqp1u",	"hk9sql"
#               ,	"1ed3vb7",	"1acelwq",	"17rq3bv",	"1eatgu4",	"1e5uhnk",	"1emgwfg",	"1eekx44",	"12h3zeg"
#               ,	"hfpg6a",	"138zhia",	"18j0x7x",	"1efegx0",	"1c2ga7m",	"1btit02",	"1epen3u",	"1aseovc"
#               ,	"1e7ghvt",	"1eehfnb",	"1e8fno4",	"1educb1",	"1eho8bt",	"1eibek7",	"1ebbwut",	"1ecftsg"
#               ,	"1ep3tg2",	"1ecct4m",	"1dksmoj",	"1e9iott",	"1eixzoj",	"1eioxwo",	"1ehb5df",	"1e7ftap"
#               ,	"1ehwq9v",	"1enqimk",	"1e8wat2",	"1eohayo",	"1e7wfjo",	"1ee6q2z",	"1eiuaeb",	"1e8exxy"
#               ,	"1eceddz",	"1ea6tj4",	"1eg01ud",	"1e7i8t1",	"1efiedr",	"1e9vbzq",	"1e89fgf",	"1e8eu5e"
#               ,	"1ekqhoj",	"1ay3rkw",	"188dgvp",	"1e9njfo",	"1ea1qrm",	"1ebztir",	"1e58j1f",	"1e8z7mq"
#               ,	"1e6qovk",	"1dn1zag",	"1e6l0h9",	"1ekm1d8",	"1dtufjo",	"1e5zuaa",	"1easrog",	"1e936gv"
#               ,	"1eeoezv",	"1eeq96c",	"1ehxc2w",	"1efhrs5",	"1e934r5",	"1eahro7",	"1efi32t",	"jlhdmk"
#               ,	"l51xx5",	"oif1so",	"youynk",	"16hsool",	"1c00oev",	"1e52dqn",	"1e61vb5",	"1e6zojw",	"jzybu2"
#               ,	"s8z1ae",	"t36hil",	"14rjmm8",	"1bcdmv8",	"1e9sjn1",	"1ed76m5",	"1efdr2p",	"hfy1v5",	"jtc2ju"
#               ,	"ju2k6h",	"qj61hg",	"xcqndf",	"z2l0l3",	"ztxqvk",	"14nauel",	"1aom4i1",	"1e4k4sj",	"1e4v2s1"
#               ,	"jlqc5w",	"jodwng",	"13p5li8",	"1b4my15",	"11brd8w",	"12leo2t",	"13002n4",	"14jqhwu",	"14nevmi"
#               ,	"1b6tja3",	"1bgsz97",	"1bp7ipi",	"1d5rv6n",	"1d8uccy",	"1db93ye",	"1dbg10f",	"1dcbxbn"
#               ,	"1dcmlpr",	"1df8twe")
# labels <- c("serious",	"misc",	"misc",	"neutral",	"serious",	"serious",	"misc",	"serious",	"neutral"
#             ,	"serious",	"serious",	"serious",	"neutral",	"serious",	"serious",	"neutral",	"misc",	"neutral"
#             ,	"serious",	"misc",	"misc",	"serious",	"serious",	"neutral",	"serious",	"serious",	"neutral"
#             ,	"serious",	"serious",	"serious",	"serious",	"serious",	"misc",	"serious",	"misc",	"serious"
#             ,	"serious",	"serious",	"misc",	"neutral",	"serious",	"misc",	"neutral",	"neutral",	"misc"
#             ,	"serious",	"serious",	"serious",	"serious",	"misc",	"serious",	"misc",	"misc",	"misc",	"misc"
#             ,	"serious",	"misc",	"serious",	"serious",	"serious",	"serious",	"serious",	"serious",	"misc"
#             ,	"misc",	"neutral",	"serious",	"misc",	"serious",	"misc",	"serious",	"serious",	"neutral"
#             ,	"serious",	"neutral",	"misc",	"misc",	"neutral",	"misc",	"serious",	"serious",	"misc",	"misc"
#             ,	"serious",	"misc",	"serious",	"neutral",	"misc",	"misc",	"misc",	"neutral",	"misc",	"serious"
#             ,	"misc",	"neutral",	"misc",	"neutral",	"misc",	"serious",	"serious",	"serious",	"misc",	"serious"
#             ,	"serious",	"serious",	"serious",	"neutral",	"serious",	"serious",	"misc",	"serious",	"serious"
#             ,	"serious",	"neutral",	"serious",	"neutral",	"neutral",	"misc",	"serious",	"misc",	"misc"
#             ,	"serious",	"serious",	"misc",	"misc",	"serious",	"neutral",	"misc",	"misc",	"misc",	"misc",	"misc"
#             ,	"misc",	"serious",	"serious",	"neutral",	"serious",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc"
#             ,	"misc",	"serious",	"misc",	"neutral",	"serious",	"serious",	"serious",	"serious",	"neutral"
#             ,	"serious",	"neutral",	"neutral",	"neutral",	"neutral",	"neutral",	"neutral",	"serious",	"neutral"
#             ,	"neutral",	"neutral",	"serious",	"neutral",	"misc",	"serious",	"serious",	"neutral",	"neutral"
#             ,	"neutral",	"neutral",	"neutral",	"serious",	"neutral",	"serious",	"neutral",	"serious",	"neutral"
#             ,	"misc",	"misc",	"serious",	"serious",	"serious",	"serious",	"neutral",	"serious",	"misc",	"serious"
#             ,	"neutral",	"misc",	"serious",	"serious",	"neutral",	"serious",	"neutral",	"neutral",	"misc"
#             ,	"neutral",	"serious",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"neutral",	"serious"
#             ,	"serious",	"serious",	"misc",	"misc",	"misc",	"misc",	"serious",	"neutral",	"neutral",	"misc"
#             ,	"misc",	"neutral",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"serious",	"serious",	"misc"
#             ,	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc"
#             ,	"misc",	"misc",	"misc",	"misc",	"misc")

post_ids_2 <- c("1e4l4mo",	"1e0aqb6",	"1e9oz6i",	"1e7lc9e",	"1e43b2h",	"1e614jl",	"17hmdeq",	"1e3d80s",	"1e8klhz",	"1ejkkc0",	"1e44ror",	"1ec6zlr",	"1eb8cho",	"1e6e7ep",	"1eepcc5",	"1el14r9",	"hfq3c6",	"1envfw4",	"1e8j4dr",	"14673o9",	"1ar4zit",	"1e555o3",	"1ea8aa5",	"1ecug9x",	"1e69doe",	"1e7bafq",	"1ehu6bs",	"1edq1k2",	"1e5cze5",	"1e4z4p9",	"1e9r5iw",	"18j6krz",	"1e8xvzu",	"1eonq7a",	"1e7ywf2",	"10uzzq8",	"1e65db4",	"1e9h8tg",	"1e9i0o3",	"16tm6d3",	"1e4549q",	"17kga00",	"1e7gogm",	"1e5rx91",	"16dm1l6",	"1eckcm3",	"1emmhfb",	"1ee9afp",	"1edwtyn",	"1dsz578",	"1e4qvay",	"1e8rm94",	"179llq2",	"12h6lbp",	"1efeo56",	"1ejx9ae",	"zpzpp2",	"1e52jlt",	"1eii4w7",	"1e4ijjv",	"1egcie6",	"1eddte0",	"1elv3gh",	"13p3ejn",	"131wman",	"1en9gtn",	"1e4w3so",	"1941781",	"1db7smd",	"1771vac",	"1ecl6ns",	"1e74iwf",	"1el8n4y",	"1e9y2f6",	"o91zei",	"189t41y",	"1agr7qq",	"1e7v2zi",	"1adwvg8",	"1e6feut",	"1ejbm0z",	"1ajrjvr",	"l0uoie",	"1e86g4u",	"17md4ht",	"1ef9q5i",	"1e4672s",	"149jdwy",	"19bpall",	"1ae2psw",	"1eedq5u",	"16o4n37",	"13bp11s",	"peyzvh",	"1397w74",	"1e3mcwf",	"1eh7i57",	"1enqb1d",	"1334oqq",	"1e9ckpb",	"1ecw41s",	"1epen3u",	"17og34a",	"1e75il7",	"tbev40",	"1e43ndd",	"nw6qyy",	"195hrnw",	"1efslbp",	"ulas8w",	"1e4adli",	"1ed2qpr",	"1e570ai",	"1efgnsq",	"1e6se4f",	"1edt76k",	"1eb32ut",	"1e8t8i4",	"1e5nr59",	"1e3zjev",	"12zz5t0",	"1bw4ty4",	"1ef4biy",	"1e52zp0",	"1brnwgp",	"1amdml1",	"1ebboxf",	"n7ubfk",	"1afpvbl",	"hk9sql",	"14iqp1u",	"1ed3vb7",	"1acelwq",	"1eatgu4",	"1emgwfg",	"1e5uhnk",	"17rq3bv",	"1eekx44",	"18j0x7x",	"138zhia",	"hfpg6a",	"12h3zeg",	"1efegx0",	"1btit02",	"1e7ghvt",	"1c2ga7m",	"1aseovc",	"1eohayo",	"1enqimk",	"1ekqhoj",	"1ekm1d8",	"1eixzoj",	"1eiuaeb",	"1eioxwo",	"1eibek7",	"1ehwq9v",	"1eho8bt",	"1ehxc2w",	"1ehb5df",	"1eg01ud",	"1efiedr",	"1efhrs5",	"1efi32t",	"1eehfnb",	"1efdr2p",	"1eeq96c",	"1eeoezv",	"1ee6q2z",	"1educb1",	"1ecftsg",	"1ecct4m",	"1eceddz",	"1ed76m5",	"1ebztir",	"1ebbwut",	"1easrog",	"1ea6tj4",	"1eahro7",	"1ea1qrm",	"1e9vbzq",	"1e9iott",	"1e9njfo",	"1e8wat2",	"1e9sjn1",	"1e936gv",	"1e934r5",	"1e8z7mq",	"1e8fno4",	"1e8exxy",	"1e8eu5e",	"oif1so",	"s8z1ae",	"1dbg10f",	"12leo2t",	"1df8twe",	"1e7ftap",	"1e6zojw",	"14jqhwu",	"1dcbxbn",	"1db93ye",	"1b4my15",	"1dtufjo",	"14rjmm8",	"1dksmoj",	"ztxqvk",	"1dcmlpr",	"ju2k6h",	"qj61hg",	"188dgvp",	"1d8uccy",	"1b6tja3",	"13002n4",	"11brd8w",	"1e4k4sj",	"1c00oev",	"13p5li8",	"1bp7ipi",	"jlqc5w",	"1bgsz97",	"1e6l0h9",	"16hsool",	"1ay3rkw",	"1e58j1f",	"hfy1v5",	"1aom4i1",	"14nevmi",	"jodwng",	"1e52dqn",	"jlhdmk",	"1d5rv6n",	"youynk",	"t36hil",	"l51xx5",	"1e6qovk",	"1e7wfjo",	"1e7i8t1",	"1e4v2s1",	"1bcdmv8",	"z2l0l3",	"1dn1zag",	"xcqndf",	"jzybu2",	"jtc2ju",	"1e5zuaa",	"1e61vb5",	"14nauel",	"1e89fgf")
length(post_ids_2)
labels_2 <- c("allegations",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"allegations",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"misc",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"misc",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"allegations",	"misc",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"misc",	"allegations",	"allegations",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"allegations",	"allegations",	"allegations",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"allegations",	"allegations",	"allegations",	"misc",	"misc",	"misc",	"misc",	"misc",	"misc",	"allegations",	"allegations",	"misc",	"allegations")
length(labels_2)
# Ensure the length of post_ids and labels are the same
if(length(post_ids_2) != length(labels_2)) {
  stop("The length of post_ids and labels must be the same.")
}

# Create a named vector for easy matching
label_vector <- setNames(labels_2, post_ids_2)
length(label_vector)
# Add the labels to the dataframe by matching post IDs
combined_data$label <- label_vector[combined_data$post_id]

# Check the dataframe to ensure labels are added
print(combined_data)

###################################################
#Train and predict
###################################################
#train data
train_data <- combined_data %>%
  filter(!is.na(label))
missing_rows <- train_data %>%
  filter(!post_id %in% post_ids_2)


# print(missing_rows)
# print(nrow(train_data))  # Should be 248
# print(length(post_ids_2)) # Should be 248

#test data
test_data <- combined_data %>% filter(is.na(label))
set.seed(123)  # Setting a seed for reproducibility
test_data <- test_data %>% sample_n(247)

train_set_sample <- train_data %>% select(combined_text, label)
test_set_sample <- test_data %>% select(combined_text, label)

#train corpus, dfm, matrix
train_corpus <- corpus(train_set_sample, text_field = "combined_text")
train_dfm <- train_corpus %>%
  tokens(remove_punct = TRUE, remove_numbers = FALSE) %>%
  tokens_tolower() %>%
  tokens_remove(expanded_stopwords) %>%
  tokens_wordstem() %>%
  dfm()
train_matrix <- as.matrix(train_dfm)
#test corpus, dfm, matrix
test_corpus <- corpus(test_set_sample, text_field = "combined_text")
test_dfm <- test_corpus %>%
  tokens(remove_punct = TRUE, remove_numbers = FALSE) %>%
  tokens_tolower() %>%
  tokens_remove(expanded_stopwords) %>%
  tokens_wordstem() %>%
  dfm()
test_matrix <- as.matrix(test_dfm)
#dependent variable is label
dependent <- train_set_sample$label

dim(train_matrix)
dim(test_matrix)
#make naive bayes model
nb_model <- naiveBayes(x = train_matrix
                       , y = as.factor(train_set_sample$label))
nb_predictions <- predict(nb_model, test_matrix)
#confusion matrix
results = data.frame(
  Predictions = nb_predictions,
  Actuals = dependent
)
results
#print accuracy results
correct_predictions <- sum(nb_predictions == dependent)
total_predictions <- length(dependent)
accuracy <- (correct_predictions / total_predictions) * 100
print(paste("Accuracy: ", accuracy, "%", sep = ""))

###############################
#svm model
###############################
labels_2 <- as.factor(labels_2)

set.seed(42)

# Create folds on the entire dataset
folds_5 <- createFolds(train_data$label, k = 5)

# SVM K-Fold Cross-Validation Function
svm_k_fold <- function(folds, data, labels, model_kernel = 'linear') {
  
  results <- lapply(folds, function(x) { 
    test_subset <- as.vector(x)  # Get test indices
    train_subset <- setdiff(seq_len(nrow(data)), test_subset)  # Get train indices
    
    # Subset the training and test data
    train_matrix <- data[train_subset, ]
    train_labels <- as.factor(labels[train_subset])
    test_matrix <- data[test_subset, ]
    test_labels <- as.factor(labels[test_subset])
    
    # Train the SVM model
    svm_mod <- e1071::svm(
      x = train_matrix,
      y = train_labels,
      type = 'C',
      kernel = model_kernel
    )
    
    pred <- predict(svm_mod, test_matrix)
    
    confusion_matrix <- confusionMatrix(data = pred, 
                                        reference = test_labels, 
                                        mode = 'prec_recall')
    
    return(confusion_matrix)
  })
  
  return(results)
}

# Run the SVM cross-validation
label_SVM_results <- svm_k_fold(folds = folds_5,
                                data = train_matrix,
                                labels = train_data$label,
                                model_kernel = 'linear')

label_SVM_results

train_labels <- train_data$label

###############################
#applying svm model to all data
###############################

full_matrix <- as.matrix(codyko_dfm)
dim(full_matrix)


full_corpus <- corpus(combined_data, text_field = "combined_text")

full_dfm <- full_corpus %>%
  tokens(remove_punct = TRUE, remove_numbers = FALSE) %>%
  tokens_tolower() %>%
  tokens_remove(expanded_stopwords) %>%
  tokens_wordstem() %>%
  dfm()

full_matrix <- as.matrix(full_dfm)

#align matricies
aligned_full_matrix <- full_matrix[, colnames(train_matrix), drop = FALSE]

#remove NAs
aligned_full_matrix[is.na(aligned_full_matrix)] <- 0

final_svm_model <- e1071::svm(x = train_matrix, y = as.factor(train_labels), type = 'C', kernel = 'linear')


#label predicrtion
full_predictions <- predict(final_svm_model, aligned_full_matrix)

combined_data$predicted_label <- full_predictions

#print(head(combined_data))

#Summary of the predictions
summary(combined_data$predicted_label)

#make a graph
# Aggregate the data by date and label
label_counts <- combined_data %>%
  group_by(date, predicted_label) %>%
  summarise(count = n()) %>%
  ungroup()

label_counts <- label_counts %>%
  complete(date, predicted_label, fill = list(count = 0))

#Plot the data
p <- ggplot(label_counts, aes(x = date, y = count, color = predicted_label, group = predicted_label)) +
  geom_line(size = 1) +  # Line for each label
  labs(title = "Allegations vs Misc Over Time",
       x = "Date",
       y = "Count",
       color = "Label") +
  scale_color_manual(values = c("allegations" = "red", "misc" = "blue")) +
  theme_minimal()

#save
ggsave("allegations_vs_misc_plot.png", plot = p, width = 10, height = 6)

###
plot_data <- label_counts %>%
  mutate(date = as.Date(date))

#Create the interactive plot directly with plotly
p_interactive <- plot_ly(plot_data, x = ~date, y = ~count, color = ~predicted_label, type = 'scatter', mode = 'lines') %>%
  layout(title = "Allegations vs Misc Over Time",
         xaxis = list(title = "Date"),
         yaxis = list(title = "Count"),
         hovermode = "x unified")


p_interactive
htmlwidgets::saveWidget(p_interactive, "allegations_vs_misc_interactive_plot.html")

library(htmlwidgets)
saveWidget(p_interactive, "interactive_plot.html", selfcontained = TRUE)
###


##############
#compare plots
###############

combined_plot_data <- sentiment_data %>%
  left_join(label_counts, by = "date")
#Plot sentiment and allegations over time
p <- ggplot(combined_plot_data) +
  geom_line(aes(x = date, y = sentiment, color = "Sentiment"), size = 1) +  # Sentiment line
  geom_line(aes(x = date, y = count, color = "Allegations"), size = 1) +  # Allegation count line
  labs(title = "Sentiment and Allegations Over Time",
       x = "Date",
       y = "Value",
       color = "Metric") +
  scale_color_manual(values = c("Sentiment" = "blue", "Allegations" = "red")) +
  theme_minimal()

print(p)

#PNG
ggsave("sentiment_vs_allegations_plot.png", plot = p)

combined_plot_data <- sentiment_data %>%
  left_join(label_counts, by = "date")

# Plot smoothed sentiment and allegations over time with increased precision
p <- ggplot(combined_plot_data) +
  geom_smooth(aes(x = date, y = sentiment, color = "Sentiment"), method = "loess", span = 0.1, se = FALSE, size = 1.8) +  # More precise smoothing for sentiment
  geom_smooth(aes(x = date, y = count, color = "Allegations"), method = "loess", span = 0.1, se = FALSE, size = 1.8) +  # More precise smoothing for allegations
  labs(title = "Sentiment and Allegations Over Time",
       x = "Date",
       y = "Value",
       color = "Metric") +
  scale_color_manual(values = c("Sentiment" = "blue", "Allegations" = "red")) +
  theme_minimal()

print(p)

# Save the plot as a PNG
ggsave("sentiment_vs_allegations_plot_precise.png", plot = p, width = 10, height = 6, dpi = 300)

################
#math
#############
correlation_data <- combined_plot_data %>%
  filter(!is.na(sentiment) & !is.na(count)) 

# Perform correlation test
correlation_test <- cor.test(correlation_data$sentiment, correlation_data$count, method = "pearson")

correlation_value <- correlation_test$estimate
p_value <- correlation_test$p.value
conf_interval <- correlation_test$conf.int

print(paste("Correlation coefficient: ", round(correlation_value, 3)))
print(paste("p-value: ", round(p_value, 5)))
print(paste("95% confidence interval: [", round(conf_interval[1], 3), ", ", round(conf_interval[2], 3), "]"))
##
library(quanteda)

##find when posts started

filtered_data <- combined_data %>%
  filter(predicted_label == "allegations") %>%  # Change "allegations" to your desired label
  arrange(date)  # Sort by date in ascending order (oldest first)


print(filtered_data$combined_text)
####
token_frequencies <- colSums(codyko_dfm)

# Sort the tokens by frequency in descending order
sorted_tokens <- sort(token_frequencies, decreasing = TRUE)

# Select the top 50 most common tokens
top_50_tokens <- head(sorted_tokens, 50)

# Print the top 50 tokens and their frequencies
print(top_50_tokens)

###################