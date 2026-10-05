library(dplyr)
library(tidyr)
library(purrr)
library(readr)
library(stringr)

setwd("~/Research/letterstringExperiment/study_philipp")

set.seed(123)  # for reproducibility

# 1. Read data -------------------------------------------------------------

### import datasets add/rm necessary cols
df_items <- read.csv("mlc_letstr_items.csv") %>%
  distinct(across(-itemid), .keep_all = TRUE) %>%
  mutate(
    transformation_category = case_when(
      transformation %in% c("pred","remove_redundant","sort","sort_group",
			    "succ","rr_interleave","rr_succ","fix_extend", "extend_sequence",
			    "fix_alphabetic_seq") 
      ~ "seen",
      transformation %in% c("rr_sort","extend_pred","fix_interleave",
			    "extend_group","extend_extend_succ",
			    "fix_pred_succ") 
      ~ "composition",
      transformation %in% c("reverse","shift","replicate") 
      ~ "novel",
      TRUE ~ NA_character_)
  ) %>%
  rename(alphabet_status = testlet) %>%
  mutate(alphabet_status = case_when(
      alphabet_status == "unseen alphabets" ~ "unseen",
      alphabet_status == "seen alphabets"   ~ "seen",
      TRUE ~ alphabet_status))


# 2. Example items for seen transformations --------------------------------

# pool of eligble examples for seen transformations
pool_seen_ex <- df_items %>%
  filter(alphabet_status == "seen", transformation_category == "seen")

# function to get pool of eligble examples for seen transformations
get_examples_seen_trans <- function(target_transformation, target_itemid, n = 5) {
  pool <- pool_seen_ex %>%
    filter(transformation == target_transformation,
           itemid != target_itemid)
  if (nrow(pool) == 0) {
    return(rep(NA_character_, n))
  }
  ids <- sample(pool$itemid,
                size = min(n, nrow(pool)),
                replace = FALSE)
  c(ids, rep(NA_character_, n - length(ids)))
}

# df of items with seen transformations
df_seen_trans <- df_items %>%
  filter(transformation_category == "seen") %>%
  rowwise() %>%
  mutate(ex_id = list(
    get_examples_seen_trans(transformation, itemid, n = 5)
  )) %>%
  ungroup() %>%
  unnest_wider(ex_id, names_sep = "_") 

# 3. Example items for novel transformations --------------------------------

# function to get pool of eligible examples for novel transformations
# these should come from the seen transformations 
# as the novel transformations themselves weren't trained
get_examples_novel_trans <- function(target_itemid, n = 5) {
  pool <- pool_seen_ex %>%
    filter(itemid != target_itemid)
  if (nrow(pool) == 0) {
    return(rep(NA_character_, n))
  }
  ids <- sample(pool$itemid,
                size = min(n, nrow(pool)),
                replace = FALSE)
  c(ids, rep(NA_character_, n - length(ids)))
}

# df of items with seen transformations
df_novel_trans <- df_items %>%
  filter(transformation_category == "novel") %>%
  rowwise() %>%
  mutate(ex_id = list(
    get_examples_novel_trans(itemid, n = 5)
  )) %>%
  ungroup() %>%
  unnest_wider(ex_id, names_sep = "_") 

# 4. Example items for compositional transformations -----------------------

# function to get list of transformations to use as training examples
# for composition transformations
get_training_trans <- function(transformation) {
  switch(
    transformation,
    "rr_sort"           = c("remove_redundant", "sort"),
    "extend_pred"       = c("extend_sequence", "pred"),
    "fix_interleave"    = c("fix_alphabetic_seq", "rr_interleave"),
    "extend_group"      = c("extend_sequence", "sort_group"),
    "extend_extend_succ"= c("extend_sequence", "succ"),
    "fix_pred_succ"     = c("fix_alphabetic_seq", "pred", "succ"),
    stop("Unknown transformation: ", transformation)
  )
}

# function to get pool of eligible examples for compositional transformations
# these should come from the seen transformations that make up the 
# compositions as the compositions themselves weren't trained
get_examples_composition_trans <- function(target_itemid, target_trans, n = 5) {
  pool <- pool_seen_ex %>%
    filter(itemid != target_itemid,
           transformation %in% target_trans)
  if (nrow(pool) == 0) {
    return(rep(NA_character_, n))
  }
  ids <- sample(pool$itemid,
                size = min(n, nrow(pool)),
                replace = FALSE)
  c(ids, rep(NA_character_, n - length(ids)))
}

# df of items with composition transformations
df_composition_trans <- df_items %>%
  filter(transformation_category == "composition") %>%
  rowwise() %>%
  mutate(ex_id = list(
    get_examples_composition_trans(itemid, get_training_trans(transformation), n = 5)
  )) %>%
  ungroup() %>%
  unnest_wider(ex_id, names_sep = "_") 

# 5. Combine dfs to form item set with ICL up to 5 examples --------------------
df_icl_items <- rbind(df_seen_trans, df_composition_trans, df_novel_trans)

write_csv(df_icl_items, "icl_letstr_items.csv")
