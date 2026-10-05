library(dplyr)
library(readr)
library(stringdist)

setwd("~/Research/letterstringExperiment/study_philipp")

import_csvs <- function(folder_path, numeric_cols = character()) {
  csv_files <- list.files(
    path        = folder_path,
    pattern     = "\\.csv$",
    full.names  = TRUE,
    ignore.case = TRUE
  )
  
  if (length(csv_files) == 0) {
    stop("No CSV files found in: ", folder_path)
  }
  
  message("Found ", length(csv_files), " CSV file(s). Importing...")
  
  combined <- lapply(csv_files, function(file) {
    col_spec <- readr::cols(
      .default = readr::col_character()
    )
    
    # Override the default character type for supplied numeric columns
    for (col in numeric_cols) {
      col_spec$cols[[col]] <- readr::col_double()
    }
    
    readr::read_csv(
      file,
      col_types = col_spec,
      show_col_types = FALSE
    )
  }) %>%
    dplyr::bind_rows() %>%
    dplyr::mutate(
      dplyr::across(
        -dplyr::all_of(numeric_cols),
        as.factor
      )
    )
  
  message(
    "Done. Combined dataset: ",
    nrow(combined),
    " rows x ",
    ncol(combined),
    " columns."
  )
  
  return(combined)
}

# ------------------------------------------------------------------
# 1.  Read the items file and force itemid to character
# ------------------------------------------------------------------
df_items <- read.csv("items/icl_letstr_selected_items.csv") %>%
  distinct(across(-itemid), .keep_all = TRUE) %>%
  mutate(itemid = as.character(itemid))          # <‑‑ force character

# ------------------------------------------------------------------
# 2.  Build the selected‑item list (still character)
# ------------------------------------------------------------------
selected_itemids <- df_items %>%
  distinct(itemid)

# ------------------------------------------------------------------
# 3.  ICL dataset
# ------------------------------------------------------------------
icl_numeric_cols <- c(
  "n_examples", "prompt_tokens", "completion_tokens",
  "reasoning_tokens", "total_tokens", "rt", "correct",
  "lev_A", "lev_B", "lev_C", "lev_D"
)

df_icl_llm_raw <- import_csvs(
  "results_icl_llms",
  numeric_cols = icl_numeric_cols
) %>%
  semi_join(selected_itemids, by = "itemid")

# Columns present in both data frames, apart from the join key
duplicate_cols <- setdiff(
  intersect(names(df_icl_llm_raw), names(df_items)),
  "itemid"
)

df_icl_llm <- df_icl_llm_raw %>%
  select(-all_of(duplicate_cols)) %>%
  left_join(df_items, by = "itemid") %>%      
  mutate(itemid = as.factor(itemid),
         response_raw = response,
         correct_old = correct) %>%
  mutate(
    response_clean = stringr::str_match(
      as.character(response),
      stringr::regex(
        "(?:^|\\s)([a-z](?![a-z])(?:\\s+[a-z](?![a-z]))*)\\s*$",
        ignore_case = TRUE
      )
    )[, 2] %>%
      stringr::str_to_lower() %>%
      stringr::str_squish()
  ) %>%
  mutate(
    correct_new = dplyr::coalesce(
      as.integer(
        stringr::str_squish(stringr::str_to_lower(as.character(response_clean))) ==
          stringr::str_squish(stringr::str_to_lower(as.character(D)))
      ),
      0L
    )
  ) %>%
  mutate(correct = correct_new)
  
# ------------------------------------------------------------------
# 5.  MLC dataset
# ------------------------------------------------------------------
mlc_numeric_cols <- c("n_perm", "query_length", "correct")
df_mlc <- import_csvs("results_mlc", numeric_cols = mlc_numeric_cols) %>%
  mutate(response = pred) %>%
  mutate(provider = "Philipp Hellwig") 
df_mlc_iclitems <- df_mlc %>%
  left_join(df_items) %>%
  filter(!is.na(itemid))
# ------------------------------------------------------------------
# 6.  Score MLC responses
# ------------------------------------------------------------------
df_mlc <- df_mlc %>%
  mutate(lev_A = stringdist(A, response, method = "lv"),
         lev_B = stringdist(B, response, method = "lv"),
         lev_C = stringdist(C, response, method = "lv"),
         lev_D = stringdist(D, response, method = "lv"))

