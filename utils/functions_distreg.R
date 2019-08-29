
library(Rfast)
library(data.table)
library(ggplot2)
library(kernlab)



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




getLandmarks = function(data, vars, n_landmarks, subset_ind = NULL){
  
  modmat_fmla = as.formula(paste('~', paste(vars, collapse = '+')))
  
  # make one modmat so we get all columns in all subsets
  X = modmat_all_levs(data = data, formula = modmat_fmla)
  
  # get group definitions with k-means
  landmarks = suppressWarnings(kmeanspp(data = X[subset_ind, ], k = n_landmarks, iter.max = 1))
  landmarks = as.matrix(landmarks$inicial.centers)

  return(list(landmarks = landmarks, X = X))
}



getFeatures = function(data, bag, train_ind, landmarks, kernel_params, weight = NULL){
  
  # calculate features for train
  phi_x = getCustomKern(X = data, Y = landmarks, kernel_params = kernel_params)
  phi_x = data.table(bag = bag, phi_x, weight)
  setnames(phi_x, c('bag', paste0('u', 1:nrow(landmarks)), 'weight'))
  
  # calculate means of embeddings
  if(!is.null(weight)){
    mu_hat_train = phi_x[train_ind == 1, lapply(.SD, weighted.mean, w = weight), .SDcols = names(phi_x)[-c(1, ncol(phi_x))], by = bag][order(bag)]
  }else{
    mu_hat_train = phi_x[train_ind == 1, lapply(.SD, mean), .SDcols = names(phi_x)[-c(1, ncol(phi_x))], by = 'bag'][order(bag)]
  }
  
  return(list(mu_hat = mu_hat_train, phi_x = phi_x[, -which(names(phi_x) == 'weight'), with = F]))
}



fitLasso = function(mu_hat, Y_bag, phi_x = NULL, nfolds = 10, family = 'gaussian', alpha = 1){
  require(glmnet)
  
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
  
  fit_lambda = cv.glmnet(x = mu_hat_mat, y = Y_bag, nfolds = nfolds, family = family, alpha = alpha)
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

  # refit to avoid shrinkage
  fit = glmnet(as.matrix(mu_hat_mat[, nonzero_ind]), y = Y_bag, lambda = 0, family = family, alpha = alpha)
  
  if(!is.null(phi_x)){
    Y_hat = predict(fit, newx = as.matrix(phi_x)[, nonzero_ind], type = 'response')
  }else{
    Y_hat = NULL
  }
  
  
  return(list(fit = fit, Y_hat = Y_hat))
}


# , make_bags_vars
# , score_bags_vars = NULL
# , regression_vars
# , outcome
# , kernel_type = 'rbf'
# , sigma = NULL
# , n_bags = NULL
# , n_landmarks = NULL
# , family = 'gaussian'
# , bagging_ind = 'surveyed'
# , train_ind = 'voterfile'
# , test_ind = 'holdout'
# , weight_col = NULL
# , landmarks = NULL
# , bags = NULL

doBasicDR = function(data, params){
  
  require(glmnet)
  require(kernlab)
  
  if(is.null(params$weight_col)){
    data[, weight := 1]
  }else{
    data[, weight := get(params$weight_col)]
  }
  
  if(is.null(params$score_bags_vars)){
    params$score_bags_vars = params$make_bags_vars
  }
  
  data[, bag := NULL]
  
  # Make bags
  if(is.null(params$bags)){
    cat(paste0(Sys.time(), "\t Making bags\n"))
    bags = getBags(data = data[get(params$bagging_ind) == 1,]
                   , vars = params$make_bags_vars
                   , n_bags = params$n_bags
                   , newdata = data[, params$score_bags_vars, with = F])
  }else{
    bags = params$bags
    n_bags = length(unique(bags$bags_newdata))
  }
  
  # assign data to bags
  data[, bag := bags$bags_newdata]
  
  # Get landmarks
  if(is.null(params$landmarks)){
    cat(paste0(Sys.time(), "\t Getting landmarks\n"))
    landmarks = getLandmarks(data = data
                             , vars = params$regression_vars
                             , n_landmarks = params$n_landmarks
                             , subset_ind = (data[, get(params$train_ind)] == 1))
  }else{
    landmarks = params$landmarks
  }
  
  # make kernel parameters
  kernel_params = getKernParams(X = landmarks$X, kernel_type = params$kernel_type, sigma = params$sigma)
  print(kernel_params)
  
  # get features
  cat(paste0(Sys.time(), "\t Making features\n"))
  features = getFeatures(data = landmarks$X
                         , bag = as.numeric(data[, bag])
                         , train_ind = as.numeric(data[, get(params$train_ind)])
                         , landmarks = landmarks$landmarks
                         , kernel_params = kernel_params
                         , weight = as.numeric(data[, weight]))
  
  
  # prep outcome
  # if weighting col is specified, then use that to get weighted mean
  if(params$outcome_family == 'multinomial'){
    #Y_svy_bag = data[get(bagging_ind) == 1, lapply(.SD, weighted.mean, w = weight), .SDcols = outcome, by = bag][order(bag)]
    Y_svy_bag = data[get(params$bagging_ind) == 1, lapply(.SD, function(s) sum(s * weight)), .SDcols = params$outcome, by = bag][order(bag)]
  }else{
    #Y_svy_bag = data[get(bagging_ind) == 1, .(y_mean = weigted.mean(get(outcome), w = weight)), bag][order(bag)]
    Y_svy_bag = data[get(params$bagging_ind) == 1, .(y_mean = sum(weight)), bag][order(bag)]
  }
  # weights might be negative and prodce negative estimates of Y, so floor at 0
  Y_svy_bag[Y_svy_bag < 0] <- 0
  
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
  
  if(params$outcome_family == 'multinomial'){
    Y_bag = as.matrix(Y_svy_bag[,-1, with = F])
  }else{
    Y_bag = Y_svy_bag$y_mean
  }
  
  cat(paste0(Sys.time(), "\t Fitting model\n"))
  
  # do basic DR
  fit = fitLasso(mu_hat = features$mu_hat
                 , Y_bag = Y_bag
                 , phi_x = features$phi_x[, -1]
                 , family = params$outcome_family
                 )
  
  # score the file
  y_hat = data.table(data.frame(fit$Y))
  setnames(y_hat, c('y_hat_dem', 'y_hat_rep', 'y_hat_oth'))
  #data[, paste0(outcome, '_hat') := as.list(data.frame(fit$Y))]
  
  # calculate mse
  mse_test = calcMSE(Y = as.numeric(unlist(data[get(params$test_ind) == 1, which(names(data) %in% params$outcome), with = F]))
                     , Y_pred = as.numeric(unlist(y_hat[which(data[,get(params$test_ind) == 1])])))
  
  return(list(data = data, fit = fit$fit, landmarks = landmarks$landmarks, bags = data$bag, y_hat = y_hat, mse_test = mse_test))
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

