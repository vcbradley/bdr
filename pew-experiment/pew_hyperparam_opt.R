library(parallel)
library(MASS)

numCores <- detectCores()
numCores


# https://medium.com/criteo-labs/hyper-parameter-optimization-algorithms-2fe447525903
n_param_sets = 64
results = data.table(sigma = exp(runif(min = -15, max = -1, n = n_param_sets))
                     , n_landmarks = round(runif(min = 10, max = 400, n = n_param_sets))
                     , n_bags = round(exp(runif(min = 2, max = 5, n = n_param_sets)))
                     , round = rep(0, n_param_sets)
                     , mse = rep(0, n_param_sets)
                     , mse_count = rep(0, n_param_sets)
)
#setnames(results, c('sigma','n_landmarks', 'n_bags', 'mse'))

#initialize counters
which_left = 1:n_param_sets
n_iter = 10

while(length(which_left) > 1){
  cat(paste(Sys.time(), "round: ", max(results[, round] + 1)), ', remaining: ',length(which_left),'\n')
  
  # double the number of iterations
  #if(this_round > 3) n_iter = 50
  
  for(p in which_left){
    cat(paste('\t param:', p, '\n'))
    
    # increment round
    results[p, round := round + 1]
    
    mse_sum = unlist(mclapply(1:n_iter, function(i){
      tryCatch(doBasicDR(data = data_recoded
                                    , bagging_vars = file_and_survey_vars
                                    , regression_vars = unique(covars[in_both == 1 | in_file == 1,]$var)
                                    #, outcome = 'y_dem'
                         , outcome = c('y_dem', 'y_rep', 'y_oth')
                         , family = 'multinomial'
                                    , n_bags = results[p, n_bags]
                                    , n_landmarks = results[p, n_landmarks]
                                    , sigma = results[p, sigma]
                                    , bagging_ind = 'surveyed'
                                    , train_ind = 'voterfile'
                                    , test_ind = 'holdout'
      )$mse_test
      , error = function(e) {
        print(e)
        return(0)})
      
      
    }, mc.cores = 5  #need this for parallelization
    ))
    results[p, mse := mse + sum(mse_sum)]
    results[p, mse_count := mse_count + sum(mse_sum > 0)]
    # 
    # for(iter in 1:n_iter){
    #   cat(iter)
    #   mse_temp = tryCatch(doBasicDR(data = data_recoded
    #                                 , bagging_vars = file_and_survey_vars
    #                                 , regression_vars = unique(covars[in_both == 1 | in_file == 1,]$var)
    #                                 , outcome = 'y_dem'
    #                                 , n_bags = results[n_results + iter, n_bags]
    #                                 , n_landmarks = results[n_results + iter, n_landmarks]
    #                                 , sigma = results[n_results + iter, sigma]
    #                                 , bagging_ind = 'surveyed'
    #                                 , train_ind = 'voterfile'
    #                                 , test_ind = 'holdout'
    #   )$mse_test
    #   , error = function(e) {
    #     print(e)
    #     return(0)})
    #   
    #   results[p, mse := mse + mse_temp]
    # }
  }
  which_left = which(results[which_left, mse/mse_count < median(mse/mse_count, na.rm = T)])
  
}


ggplot(results, aes(x = log(sigma), y = mse/mse_count, color = round)) + geom_point()
ggplot(results, aes(x = n_landmarks, y = mse/mse_count, color = round)) + geom_point()
ggplot(results, aes(x = n_bags, y = mse/mse_count, color = round)) + geom_point()

# best 
results[round == max(round)]


