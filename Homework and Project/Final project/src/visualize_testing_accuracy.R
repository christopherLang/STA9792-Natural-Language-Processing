library(tidyverse)

dat <- read_csv('diagnostics/knn_classifier-tfidf-test_result.csv')

dat <- (
  dat %>%
    mutate(metric = as.factor(metric),
           ngram_param = factor(ngram_param, c("1--1", "2--2", "3--3",
                                               "1--3", "2--3"),
                                c('unigram only', 'bigram only',
                                  'trigram only', 'unigram, bigram, trigram',
                                  'bigram, trigram'), ordered = T))
)

ggplot(dat) + aes(k, testing_acc) + geom_line() +
  facet_grid(metric ~ ngram_param) +
  labs(x = 'k', y = 'Accuracy (Testing Set)') +
  scale_y_continuous(labels = scales::percent) +
  theme_bw() +
  theme(strip.text = element_text(size = 12),
        axis.text = element_text(size = 10),
        axis.title = element_text(size = 11, face = 'bold'))

ggsave('figures/knn_testing_accuracy_plots.png')

best <- dat %>% filter(testing_acc == max(testing_acc))
