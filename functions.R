
library(Rfast)
library(data.table)
library(ggplot2)
library(kernlab)


calcMSE = function(Y, Y_pred){
  mean((Y_pred - Y)^2, na.rm = T)
}

#Utility function for creating a design matrix with all levels of factor variables included, rather than omitting a reference level
#as one should generally do when passing a design matrix to glmnet/cv.glmnet for a lasso (or elastic net) model
modmat_all_levs=function(formula, data, sparse = F) UseMethod("modmat_all_levs",data)

modmat_all_levs.data.frame=function(formula, data, sparse = F){
  #data.frame method: data should be a data.frame containing the variables referred to in formula, which determines the model matrix
  terms_data=data.frame(lapply(data[,all.vars(formula),drop=F],function(x)if (is.character(x)) as.factor(x) else x)) #convert character to factor #need drop=F to maintain a dataframe if only one factor!
  
  if(sparse){
    require(Matrix)
    sparse.model.matrix(formula,terms_data,contrasts.arg = lapply(terms_data[,sapply(terms_data,function(x)is.factor(x)),drop=F], contrasts, contrasts=FALSE)) 
  }else{
    model.matrix(formula,terms_data,contrasts.arg = lapply(terms_data[,sapply(terms_data,function(x)is.factor(x)),drop=F], contrasts, contrasts=FALSE)) 
  }
  
}
modmat_all_levs.data.table=function(formula, data, sparse = F){
  #data.table method: data should be a data.table containing the variables referred to in formula, which determines the model matrix
  require(data.table)
  terms_data=setDT(lapply(data[,all.vars(formula),with=F],function(x)if (is.character(x)) as.factor(x) else x))
  
  if(sparse){
    require(Matrix)
    sparse.model.matrix(formula,terms_data,contrasts.arg = lapply(terms_data[,sapply(terms_data,function(x)is.factor(x)),with=F,drop=F], contrasts, contrasts=FALSE)) #need drop=F to maintain a dataframe if only one factor!
  }else{
    model.matrix(formula,terms_data,contrasts.arg = lapply(terms_data[,sapply(terms_data,function(x)is.factor(x)),with=F,drop=F], contrasts, contrasts=FALSE)) #need drop=F to maintain a dataframe if only one factor!
  }
}


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
    data[, matched := 1]
  }
  
  data[, voterfile := ifelse(holdout == 1 | (surveyed == 1 & matched == 0), 0, 1)]
  
  return(list(data = data, holdout_ind = holdout_ind, survey_ind = survey_ind, matched_ind = matched_ind))
}



# function fixed from LICORS package
kmeanspp = function(data, k = 2, start = "random", iter.max = 100, nstart = 10, 
                    ...){
  kk <- k
  if (length(dim(data)) == 0) {
    data <- matrix(data, ncol = 1)
  }
  else {
    data <- cbind(data)
  }
  num.samples <- nrow(data)
  ndim <- ncol(data)
  data.avg <- colMeans(data)
  data.cov <- cov(data)
  out <- list()
  out$tot.withinss <- Inf
  for (restart in seq_len(nstart)) {
    center_ids <- rep(0, length = kk)
    if (start == "random") {
      center_ids[1:2] = sample.int(num.samples, 1)
    }
    else if (start == "normal") {
      center_ids[1:2] = which.min(dmvnorm(data, mean = data.avg, 
                                          sigma = data.cov))
    }
    else {
      center_ids[1:2] = start
    }
    for (ii in 2:kk) {
      dists <- dista(x = data, xnew = data[center_ids, ], square = TRUE)
      probs <- apply(dists, 2, min)
      # if (ndim == 1) {
      #   dists <- apply(cbind(data[center_ids, ]), 1, 
      #                  function(center) {
      #                    
      #                    colSums(apply(data, 1, function(x) (x - center)^2))  #line changed to correctly calc distance
      #                  })
      # }
      # else {
      #   dists <- apply(data[center_ids, ], 1, function(center) {
      #     colSums(apply(data, 1, function(x) (x - center)^2))  #line changed to correctly calc distance
      #   })
      # }
      # probs <- apply(dists, 1, min)
      probs[center_ids] <- 0
      #center_ids[ii] <- sample.int(num.samples, 1, prob = probs)
      center_ids[ii] <- sample(c(1:num.samples)[probs > 0], prob = probs[probs > 0], size = 1)
    }
    #cat(paste('n unique: ', nrow(unique(data[center_ids, ]))))
    #cat('\n')
    
    tmp.out <- kmeans(data, centers = data[center_ids, ], 
                      iter.max = iter.max, ...)
    tmp.out$inicial.centers <- data[center_ids, ]
    if (tmp.out$tot.withinss < out$tot.withinss) {
      out <- tmp.out
    }
  }
  invisible(out)
}



getBags = function(data = NULL, vars, n_bags = NULL, newdata = NULL, bags = NULL, max_attempts = 20){
  # get modmats
  bags_modmat_fmla = as.formula(paste('~', paste(vars, collapse = '+')))
  if(!is.null(data)){
    X = modmat_all_levs(data = data, formula = bags_modmat_fmla)
  }
  
  
  # make bags
  if(is.null(bags)){
    # get bags with k-means++ - wrapper so that it re-initializes if it fails the first time
    #running_landmarks = TRUE
    bags = simpleError('start')
    counter = 1
    
    while('error' %in% class(bags) & counter <= max_attempts){
      start = sample.int(nrow(X), size = 1)
      #cat(paste0("attempt ", counter, ": ", start, "\n"))
      
      bags = tryCatch({
        kmeanspp(data = X, k = n_bags, start = start)  # if running landmarks, we just need the initial points
      }, error = function(e){
        cat(paste(e))
        return(e)
      })
      
      counter = counter + 1
    }
  }
  
  if(!is.null(newdata)){
    # drop variables that we don't have in new data
    vars_new = vars[vars %in% names(newdata)]
    bags_modmat_fmla_new = as.formula(paste('~', paste(vars_new, collapse = '+')))
    
    # create new modmat
    X_new = modmat_all_levs(data = newdata, formula = bags_modmat_fmla_new)
    
    # modify bags - drop vars not in new data
    bags$centers_full = bags$centers
    
    bag_vars_new = which(colnames(bags$centers) %in% colnames(X_new))
    bags$centers = bags$centers[, bag_vars_new]
    
    bags_newdata = predict.kmeans(object = bags, newdata = X_new, method = 'classes')
  }else{
    bags_newdata = NULL
  }
  
  return(list(bag_fit = bags, bags_newdata = bags_newdata))
}





#### Code implementing kmeans++ initialization manually
# 
# n_bags = 30
# centers = rep(NA, n_bags)
# centers[1] = sample.int(n = nrow(X), size = 1)
# 
# d = as.matrix(dist(X))
# dim(d)
# 
# head(d)
# 
# for(i in 2:n_bags){
#   last = i - 1
#   points_left = c(1:nrow(X))[-centers[1:last]]
#   if(i == 2){
#     min_dist = d[centers[1:last], points_left]
#   }else{
#     min_dist = apply(d[centers[1:last], points_left], 2, min)
#   }
# 
#   centers[i] = sample(points_left
#                       , prob = min_dist^2  #choose proportionally to
#                       , size = 1)
# }
# 
# nrow(unique(X[centers,]))
# 
# which(apply(X, 1, function(x) identical(x, X[40,])))
# 
# which(d[40,] == 0)


getLandmarks = function(data, vars, n_landmarks, subset_ind = NULL){
  
  modmat_fmla = as.formula(paste('~', paste(vars, collapse = '+')))
  
  # make one modmat so we get all columns in all subsets
  X = modmat_all_levs(data = data, formula = modmat_fmla)
  
  # get group definitions with k-means
  landmarks = kmeanspp(data = X[subset_ind, ], k = n_landmarks, iter.max = 1)
  landmarks = as.matrix(landmarks$inicial.centers)

  return(list(landmarks = landmarks, X = X))
}



# https://stats.stackexchange.com/questions/12623/predicting-cluster-of-a-new-object-with-kmeans-in-r/188628
predict.kmeans <- function(object,
                           newdata,
                           method = c("centers", "classes")) {
  method <- match.arg(method)
  
  centers <- object$centers
  ss_by_center <- apply(centers, 1, function(x) {
    colSums((t(newdata) - x) ^ 2)
  })
  best_clusters <- apply(ss_by_center, 1, which.min)
  
  if (method == "centers") {
    centers[best_clusters, ]
  } else {
    best_clusters
  }
}


getFeatures = function(data, bag, train_ind, landmarks, sigma){
  # create kernel
  rbf = rbfdot(sigma = sigma)
  
  # calculate features for train
  phi_x = kernelMatrix(rbf, x = as.matrix(data), y = landmarks)
  phi_x = data.table(bag = bag, phi_x)
  setnames(phi_x, c('bag', paste0('u', 1:nrow(landmarks))))
  
  # calculate means of embeddings
  mu_hat_train = phi_x[train_ind == 1, lapply(.SD, mean), .SDcols = names(phi_x)[-1], by = 'bag'][order(bag)]
  
  return(list(mu_hat = mu_hat_train, phi_x = phi_x))
}

fitLasso = function(mu_hat, Y_bag, phi_x = NULL, nfolds = 10, family = 'gaussian'){
  
  # fit once to get non-zero coefs
  # cat(dim(mu_hat))
  # cat('\nY_bag:')
  # cat(length(Y_bag))
  # cat('\n')
  
  #cap nfolds
  nfolds = min(3, nfolds)
  
  if('data.table' %in% class(mu_hat)){
    mu_hat_mat = as.matrix(mu_hat[, -1, with = F]) # drop bags col
  }else{
    mu_hat_mat = mu_hat
  }
  
  fit_lambda = cv.glmnet(x = mu_hat_mat, y = Y_bag, nfolds = nfolds, family = family)
  if(family == 'gaussian'){
    nonzero_ind = which(coef(fit_lambda, s = 'lambda.min')[-1] != 0)  # drop intercept term
  }else{
    nonzero_ind = sort(unique(unlist(lapply(coef(fit_lambda, s = 'lambda.min'), function(c){
      which(c[-1] != 0)
    }))))
  }
  
  # use all vars again if we didn't find any sig ones the first time
  if(length(nonzero_ind) == 0){
    nonzero_ind = 1:ncol(mu_hat_mat)
  }
  
  
  # cat(dim(mu_hat))
  # cat('\n')
  # cat(dim(mu_hat_mat))
  # 
  #  cat(nonzero_ind)
  #  cat('\n')
  
  # refit to avoid shrinkage
  fit = glmnet(as.matrix(mu_hat_mat[, nonzero_ind]), y = Y_bag, lambda = 0, family = family)
  
  if(!is.null(phi_x)){
    Y_hat = predict(fit, newx = as.matrix(phi_x)[, nonzero_ind], type = 'response')
  }else{
    Y_hat = NULL
  }
  
  
  return(list(fit = fit, Y_hat = Y_hat))
}



doBasicDR = function(data
                     , make_bags_vars
                     , score_bags_vars
                     , regression_vars
                     , outcome
                     , n_bags
                     , n_landmarks
                     , sigma
                     , family = 'gaussian'
                     , bagging_ind = 'surveyed'
                     , train_ind = 'voterfile'
                     , test_ind = 'holdout'
                     , weight_col = NULL
){
  
  require(glmnet)
  require(kernlab)
  
  if(is.null(weight_col)){
    pew_data[, weight := rep(1, nrow(pew_data))]
  }else{
    pew_data[, weight := get(weight_col)]
  }
  
  
  data[, bag := NULL]
  
  # Make bags
  cat(paste0(Sys.time(), "\t Making bags\n"))
  bags = getBags(data = data[get(bagging_ind) == 1,]
                 , vars = make_bags_vars
                 , n_bags = n_bags
                 , newdata = data[, score_bags_vars, with = F])
  
  # assign data to bags
  data[, bag := bags$bags_newdata]
  
  # Get landmarks
  cat(paste0(Sys.time(), "\t Getting landmarks\n"))
  landmarks = getLandmarks(data = data
                           , vars = regression_vars
                           , n_landmarks = n_landmarks
                           , subset_ind = (data[, get(train_ind)] == 1))
  
  # Make matricies for feature embedding
  X_file = landmarks$X[data[, get(train_ind)] == 1, ]
  X_file_holdout = landmarks$X[data[, get(test_ind)] == 1, ]
  
  # get features
  cat(paste0(Sys.time(), "\t Making features\n"))
  features = getFeatures(data = landmarks$X
                         , bag = as.numeric(data[, bag])
                         , train_ind = as.numeric(data[, get(train_ind)])
                         , landmarks = landmarks$landmarks
                         , sigma = sigma)
  
  
  # prep outcome
  # if weighting col is specified, then use that to get weighted mean
  if(family == 'multinomial'){
    Y_svy_bag = data[get(bagging_ind) == 1, lapply(.SD, weighted.mean, w = weight), .SDcols = outcome, by = bag][order(bag)]
  }else{
    Y_svy_bag = data[get(bagging_ind) == 1, .(y_mean = weigted.mean(get(outcome), w = weight)), bag][order(bag)]
  }
  
  # make sure Y has all bags
  if(nrow(Y_svy_bag) < n_bags){
    warning("not all bags in survey\n")
    Y_svy_bag = merge(data.table(bag = 1:n_bags), Y_svy_bag, all.x = T, by = 'bag')
    Y_svy_bag[is.na(Y_svy_bag)] <- 0
  }
  
  # drop y obs not in voterfile bags
  if(nrow(features$mu_hat) != n_bags){
    warning("not all bags in voterfile\n")
    Y_svy_bag = Y_svy_bag[-which(!Y_svy_bag$bag %in% features$mu_hat$bag), ]
  }
  
  if(family == 'multinomial'){
    Y_bag = as.matrix(Y_svy_bag[,-1, with = F])
  }else{
    Y_bag = Y_svy_bag$y_mean
  }
  
  cat(paste0(Sys.time(), "\t Fitting model\n"))
  
  # do basic DR
  fit = fitLasso(mu_hat = features$mu_hat
                 , Y_bag = Y_bag
                 , phi_x = features$phi_x[, -1]
                 , family = family
                 )
  
  # score the file
  data[, paste0(outcome, '_hat') := as.list(data.frame(fit$Y))]
  
  # calculate mse
  mse_test = calcMSE(Y = as.numeric(unlist(data[get(test_ind) == 1, which(names(data) %in% outcome), with = F]))
                     , Y_pred = as.numeric(unlist(data[get(test_ind) == 1, which(names(data) %in% paste0(outcome, '_hat')), with = F])))
  
  return(list(data = data, fit = fit$fit, landmarks = landmarks$landmarks, bags = data$bag, y_hat = data[, which(names(data) %in% paste0(outcome, '_hat')), with = F], mse_test = mse_test))
}



doDecilePlot = function(data, score_name, title = NULL){
  # get score deciles
  pew_data[, paste0(score_name, '_dec') := cut(get(score_name)
                                               , breaks = quantile(get(score_name), probs = seq(0,1,0.1)) + (c(0:10) * .Machine$double.eps)  # add jitter
                                               , labels = 1:10, include.lowest = T)]
  
  # plot avg outcome by score decile
  plot = ggplot(pew_data[, .(pct_outcome = mean(get(score_name))), by = get(paste0(score_name, '_dec'))]
         , aes(x = get, y = pct_outcome)) + 
    geom_bar(stat = 'identity') +
    xlab("score decile") + ylab(paste0("avg ", score_name)) +
    ylim(0, 1)

  if(!is.null(title)){
    plot = plot + ggtitle(title)
  }
    
  return(plot)
}

### replacing with nearPD from the Matrix package
# ridgeMat = function(origMat){
#   cholStatus <- try(u <- chol(origMat), silent = TRUE)
#   cholError <- ifelse(class(cholStatus) == "try-error", TRUE, FALSE)
#   
#   newMat <- origMat
#   iter <- 0
#   while (cholError) {
#     
#     iter <- iter + 1
#     cat("iteration ", iter, "\n")
#     
#     # replace -ve eigen values with small +ve number
#     newEig <- eigen(newMat)
#     newEig2 <- newEig$values + 2 *abs(min(newEig$values))# + 1e-10  # add in the min eigenvalue
#     #newEig2 <- ifelse(newEig$values < 1e-10, 1e-10, newEig$values)
#     
#     # create modified matrix eqn 5 from Brissette et al 2007, inv = transp for
#     # eig vectors
#     newMat <- newEig$vectors %*% diag(newEig2) %*% t(newEig$vectors)
#     
#     # normalize modified matrix eqn 6 from Brissette et al 2007
#     newMat <- newMat/sqrt(diag(newMat) %*% t(diag(newMat)))
#     
#     # try chol again
#     cholStatus <- try(u <- chol(newMat), silent = TRUE)
#     cholError <- ifelse(class(cholStatus) == "try-error", TRUE, FALSE)
#   }
#   return(newMat)
# }


doKMM = function(X_trn, X_tst, B = 1, sigma = 1, kernel_type = 'rbf'){
  require(Matrix)
  
  n_trn = nrow(X_trn)
  n_tst = nrow(X_tst)
  
  eps = B/sqrt(n_trn)  # set epsilon based on B and suggested value from Gretton chapter; this constraint ensures that  Beta * the training dist is close to a probability dist
  
  # use RBF kernel for now
  #rbf1 = rbfdot(sigma = sigma)
  #K = kernelMatrix(rbf1, x = X_trn)
  
  kern = vanilladot()
  K = kernelMatrix(kern, x = X_trn)
  
  # fix to make sure we can use Cholesky decomp
  newK = nearPD(K)$mat
  #chol(newK)
  
  kappa = kernelMatrix(kern, x = X_trn, y = X_tst)
  kappa = (n_trn/n_tst) * rowSums(kappa)
  
  G = as.matrix(rbind(- rep(1, n_trn)
                      , rep(1, n_trn)
                      , - diag(n_trn)
                      , diag(n_trn)
  ))
  h = c(- n_trn * (1 + eps)
        , n_trn * (1 - eps)
        , - B * rep(1, n_trn)
        , rep(0, n_trn)
  )
  
  sol = solve.QP(Dmat = newK, dvec = kappa, Amat = t(G), bvec = h)
  
  return(sol)
}



getWeights = function(data, vars, train_ind, target_ind, weight_col = NULL, B = 5, sigma = 0.1){
  
  fmla = as.formula(paste0('~-1', paste(vars, collapse = '+')))
  X = model.matrix(object = fmla, data = data)  #not all levels
  
  # if weights are supplied, multiply them through the observations of X
  if(!is.null(weight_col)){
    X = diag(data[, get(weight_col)]) %*% X
  }
  
  X_train = X[which(data[, get(train_ind)] == 1),]    # data to weight
  X_target = X[which(data[, get(target_ind)] == 1),]  # target
  
  w_matched = doKMM(X_trn = X_train, X_tst = X_target, sigma = sigma, B = B)  # the smaller sigma is, the more weighting that happens 
  
  # calculate weights
  w_matched$weights = (nrow(X_matched)/sum(w_matched$solution)) * w_matched$solution
  
  return(weights = w_matched$weights)
}
