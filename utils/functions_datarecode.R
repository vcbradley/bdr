#function to recode Pew data
doPewRecode = function(data){
  
  is_march_data = ('ftcalldt' %in% names(data))
  
  # mode type
  sample_col = ifelse(is_march_data, 'samptype', 'sample')
  data[, demo_mode := ifelse(grepl('Cell', get(sample_col)), 'cell', 'landline')]
  
  # Date and month called
  if(is_march_data){
    data[, date_called := as.Date(as.numeric(substr(ftcalldt,1,6)) - min(as.numeric(substr(data_march18$ftcalldt,1,6))), origin = as.Date('2018-03-07'))]
    data[, month_called := month(date_called)]
  }else{
    data[, date_called := as.Date.character(int_date, format = '%y%m%d')]
    data[, month_called := month(as.Date.character(int_date, format = '%y%m%d'))]
  }
  
  # age
  data[, age_num := as.numeric(as.character(age))]
  
  data[, demo_age_bucket := cases('99-DNK/refused' = is.na(age_num)
                                  , '01-Under 30' = age_num < 30
                                  , '02-30to39' = age_num >= 30 & age_num < 40
                                  , '03-40to49' = age_num >= 40 & age_num < 50
                                  , '04-50to59' = age_num >= 50 & age_num < 60
                                  , '05-60to69' = age_num >= 60 & age_num < 70
                                  , '06-Over 70' = age_num >= 70
                                  , '99-DNK/refused' = TRUE 
  )]
  data[is.na(age_num), demo_age_bucket := '99-DNK/refused']
  
  # Sex
  data[, demo_sex := ifelse(sex == 'Female', '01-female', '02-male')]
  data[, .N, .(sex, demo_sex)]
  
  # Education
  data[, demo_education := cases('01-postgrad' = grepl("Postgraduate|Some postgraduate", educ)
                                 , '02-bach' = grepl("Four year", educ)
                                 , '03-assoc' = grepl("Two year associate|Some college", educ)
                                 , '04-HS' = grepl("High school graduate", educ)
                                 , '05-less than HS' = grepl("Less than high school|High school incomplete", educ)
                                 , '99-DK/refused' = TRUE
  )]
  data[, .N, .(educ, demo_education)]
  
  # Race/ethnicity
  data[, demo_hispanic := ifelse(hisp == 'Yes', '01-Y', '02-N')]
  data[, demo_race := cases('W' = (racecmb == 'White')
                            , 'B' = (racecmb == 'Black or African-American')
                            , 'A' = (racecmb == 'Asian or Asian-American')
                            , 'O' = TRUE
  )]
  data[, .N, .(demo_race, racecmb)]
  
  
  # religion
  data[, demo_relig := cases('01-protestant' = grepl("Protestant|Christian", relig)
                             , '02-catholic' = grepl("Catholic", relig)
                             , '03-athiest' = grepl("Athiest|Agnostic", relig)
                             , '04-jewish' = grepl("Jewish", relig)
                             , '05-muslim' = grepl("Muslim", relig)
                             , '06-LDS' = grepl("Mormon", relig)
                             , '07-other' = TRUE
  )]
  data[, .N, .(relig, demo_relig)]
  
  ## income
  data[, demo_income := cases('01-under10k' = (income == 'Less than $10,000')
                              , '02-10to20k' = (income == '10 to under $20,000')
                              , '03-20to30k' = (income == '20 to under $30,000')
                              , '04-30to40k' = (income == '30 to under $40,000')
                              , '05-40to50k' = (income == '40 to under $50,000')
                              , '06-50to75k' = (income == '50 to under $75,000')
                              , '07-75to100k' = (income == '75 to under $100,000')
                              , '08-100to150k' = (income == '100 to under $150,000 [OR]')
                              , '09-over150k' = (income == '$150,000 or more')
                              , '99-DK/refused' = TRUE
  )]
  data[, .N, .(demo_income, income)]
  
  
  ## household size
  data[, demo_hhsize := hh1]
  
  ## phone type
  if(is_march_data){
    data[, demo_phonetype := cases(
      '01-Both' = ((samptype == 'Cell Phone' & LL == 'Landline') | (samptype == 'Landline' & l1 == 'Yes, have cell phone'))
      , '02-LL only' = (samptype == 'Landline' & l1 != 'Yes, have cell phone')
      , '03-Cell only' = (samptype == 'Cell Phone' & LL == 'No landline')
      , '99-DNK' = TRUE
    )]
    data[, .N, .(demo_phonetype,samptype, l1, l1a, c1, LL)][order(demo_phonetype)]
  }else{
    data[, demo_phonetype := cases(
      '01-Both' = ((sample == 'Cell phone' & ll == 'Landline') | (sample == 'Landline' & ql1 == 'Yes, have cell phone'))
      , '02-LL only' = (sample == 'Landline' & ql1 != 'Yes, have cell phone')
      , '03-Cell only' = (sample == 'Cell phone' & ll == 'No landline')
      , '99-DNK' = TRUE
    )]
    data[, .N, .(demo_phonetype, sample, ql1, ql1a, qc1, ll)]
  }
  
  
  ## state
  data[, demo_state := as.character(sstate)]
  
  ## region
  data[, demo_region := as.character(scregion)]
  
  ## county population density quintile
  data[, demo_pop_density := cases(
    '01-Lowest' = sdensity == 'Lowest'
    , '02' = sdensity == 2
    , '03' = sdensity == 3
    , '04' = sdensity == 4
    , '05-Highest' = sdensity == 'Highest'
  )]
  data[, .N, .(sdensity, demo_pop_density)]
  
  ## Registered
  data[, demo_reg := cases(
    '01-Yes' = grepl('ABSOLUTELY CERTAIN', reg)
    , '02-Probably' = grepl('PROBABLY', reg)
    , '03-No' = grepl('NOT', reg)
    ,'99-DNK/refused' = TRUE
  )]
  data[, .N, reg]
  
  
  # Party
  data[, party := as.character(party)]
  data[, partyln := as.character(partyln)]
  data[is.na(partyln), partyln := 'none']
  data[, demo_party := cases(
    '01-Dem' = (party == 'Democrat')
    , '02-Rep' = (party == 'Republican')
    , '03-Lean Dem' = (partyln == 'Democrat')
    , '04-Lean Rep' = (partyln == 'Republican')
    , '05-Ind' = (party == 'Independent' | party == '(VOL) Other party')  & partyln == "(VOL) Other/Don't know/Refused"
    , '06-None' = party == '(VOL) No preference' & partyln == "(VOL) Other/Don't know/Refused"
    , '99-DK/refused' = TRUE
  )]
  
  # Ideology
  data[, demo_ideology := cases(
    '01-Very liberal' = ideo == 'Very liberal'
    , '02-Liberal' = ideo == 'Liberal [OR]'
    , '03-Moderate' = ideo == 'Moderate'
    , '04-Conservative' = ideo == 'Conservative'
    , '05-Very conservative' = ideo == 'Very conservative'
    , '99-DK/refused' = TRUE
  )]
  
  
  ## Support
  strong_col = ifelse(max(data$month_called) == 9, 'q7', ifelse(max(data$month_called) == 6, 'q20', 'q8'))
  lean_col = ifelse(max(data$month_called) == 9, 'q8', ifelse(max(data$month_called) == 6, 'q21', 'q9'))
  
  data[, qsupport := NULL]
  data[get(strong_col) == "Democratic Party's candidate" | get(lean_col) == "Democratic Party's candidate", qsupport := '1-D']
  data[get(strong_col) == "Republican Party's candidate" | get(lean_col) == "Republican Party's candidate", qsupport := '2-R']
  data[get(lean_col) == "(VOL) Other", qsupport := '3-O']
  data[is.na(qsupport), qsupport := '4-DK/R']
  
  data[, .N, .(qsupport, get(strong_col), get(lean_col))]
  
  data[, y_dem := as.numeric(qsupport == '1-D')]
  data[, y_rep := as.numeric(qsupport == '2-R')]
  data[, y_oth := as.numeric(qsupport == '3-O' | qsupport == '4-DK/R')]
  
  X = data[, grepl('demo_|month_called|y_|age', names(data)), with = F]
  return(X)
}

# data_recoded[, demo_age_bucket := NULL]
# data_recoded[, demo_age_bucket := as.character(cases(
#                                 '01-Under 30' = (age_num < 30)
#                                 , '02-30to39' = (age_num >= 30 & age_num < 40)
#                                 , '03-40to49' = (age_num >= 40 & age_num < 50)
#                                 , '04-50to59' = (age_num >= 50 & age_num < 60)
#                                 , '05-60to69' = (age_num >= 60 & age_num < 70)
#                                 , '06-Over 70' = (age_num >= 70)
#                                 , '99-DNK/refused' = TRUE
#                                 ))]
# data_recoded[, class(demo_age_bucket)]
# 
# data_recoded[, .N, demo_age_bucket]




getTestTrain = function(data, n_holdout, n_surveyed, n_matched, p_surveyed = NULL, p_matched = NULL){
  if(is.null(p_surveyed)){
    p_surveyed = rep(1/nrow(data), n = nrow(data))
  }
  if(is.null(p_matched)){
    p_matched = rep(1/n_surveyed, n = nrow(data))
  }
  
  # holdout set
  holdout_ind = sample.int(n = nrow(data), size = n_holdout, replace = F)
  
  data[, holdout := NULL]
  data[holdout_ind, holdout := 1]
  data[-holdout_ind, holdout := 0]
  
  # select indicies of those surveyed OF THOSE NOT HELDOUT
  survey_ind = sample(x = setdiff(1:nrow(data), holdout_ind), size = n_surveyed, replace = F, prob = p_surveyed[-holdout_ind])
  
  # make indicators
  data[, surveyed := NULL]
  data[survey_ind, surveyed := 1]
  data[-survey_ind, surveyed := 0]
  
  # select matched
  matched_ind = sample(survey_ind, size = n_matched, replace = F, prob = p_matched[survey_ind])
  
  data[, matched := NULL]
  if(length(matched_ind) > 0){
    data[matched_ind, matched := 1]
    data[-matched_ind, matched := 0]
  }else{
    data[, matched := 0]
  }
  
  data[, voterfile := ifelse(holdout == 1 | (surveyed == 1 & matched == 0), 0, 1)]
  
  return(list(data = data, holdout_ind = holdout_ind, survey_ind = survey_ind, matched_ind = matched_ind))
}
