# Get the child estimated AoA using logistic regression passing through 0.50.
# Automatically pulls data using wordbankr.
# Code from: https://mikabr.io/aoa-prediction/aoa_estimation.html

# Step 1: get data.
library(plyr)
library(dplyr)
library(tidyr)
library(purrr)
library(wordbankr)
admins <- get_administration_data() %>%
  select(data_id, age, language, form)

items <- get_item_data() %>%
  mutate(num_item_id = as.numeric(substr(item_id, 6, nchar(item_id))),
         definition = tolower(definition))

# Note: can use WS (words and sentences) or WG (words and gestures).
words <- items %>%
  filter(type == "word", !is.na(uni_lemma), form == "WS")

invalid_uni_lemmas <- words %>%
  group_by(uni_lemma) %>%
  filter(n() > 1,
         length(unique(lexical_class)) > 1) %>%
  arrange(language, uni_lemma)

get_inst_data <- function(inst_items) {
  inst_language <- unique(inst_items$language)
  inst_form <- unique(inst_items$form)
  inst_admins <- filter(admins, language == inst_language, form == inst_form)
  get_instrument_data(language = inst_language,
                      form = inst_form,
                      items = inst_items$item_id,
                      administrations = inst_admins,
                      iteminfo = inst_items) %>%
    filter(!is.na(age)) %>%
    mutate(produces = !is.na(value) & value == "produces",
           understands = !is.na(value) & (value == "understands" | value == "produces")) %>%
    select(-value) %>%
    gather(measure, value, produces, understands) %>%
    mutate(language = inst_language,
           form = inst_form)
}

items_by_inst <- split(words, paste(words$language, words$form, sep = "_"))
raw_data <- map(items_by_inst, get_inst_data)

# Step 2: fit models.
library(boot)
fit_inst_measure_uni <- function(inst_measure_uni_data) {
  ages <- min(inst_measure_uni_data$age):max(inst_measure_uni_data$age)
  model <- glm(cbind(num_true, num_false) ~ age, family = "binomial",
               data = inst_measure_uni_data)
  fit <- predict(model, newdata = data.frame(age = ages), se.fit = TRUE)
  aoa <- -model$coefficients[["(Intercept)"]] / model$coefficients[["age"]]
  constants <- inst_measure_uni_data %>%
    select(language, form, measure, lexical_category, lexical_class, uni_lemma,
           words) %>%
    distinct()
  props <- inst_measure_uni_data %>%
    ungroup() %>%
    select(age, prop)
  data.frame(age = ages, fit_prop = inv.logit(fit$fit), fit_se = fit$se.fit,
             aoa = aoa, language = constants$language, form = constants$form,
             measure = constants$measure,
             lexical_category = constants$lexical_category,
             lexical_class = constants$lexical_class,
             uni_lemma = constants$uni_lemma, words = constants$words) %>%
    left_join(props)
}

fit_inst_measure <- function(inst_measure_data) {
  inst_measure_by_uni <- inst_measure_data %>%
    group_by(language, form, measure,
             lexical_category, lexical_class, uni_lemma,
             age, data_id) %>%
    summarise(uni_value = any(value),
              words = paste(definition, collapse = ", ")) %>%
    group_by(language, form, measure,
             lexical_category, lexical_class, uni_lemma, words,
             age) %>%
    summarise(num_true = sum(uni_value, na.rm = TRUE),
              num_false = n() - num_true,
              prop = mean(uni_value, na.rm = TRUE))
  inst_measure_by_uni %>%
    split(paste(.$lexical_category, .$lexical_class, .$uni_lemma)) %>%
    map(fit_inst_measure_uni) %>%
    bind_rows()
}

fit_inst <- function(inst_data) {
  inst_data %>%
    split(.$measure) %>%
    map(fit_inst_measure) %>%
    bind_rows()
}

prop_data <- map(raw_data, fit_inst)
all_prop_data <- bind_rows(prop_data)

Encoding(all_prop_data$words) <- "UTF-8"
all_prop_data$word_bytes <- bytes(all_prop_data$words)

# Step 3: save.
write.table(all_prop_data, file='data/all_prop_data_WS_hex.tsv', quote=FALSE, sep='\t',  fileEncoding="UTF-8")