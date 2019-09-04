
library(doMC)
library(parallel)

detectCores(all.tests = FALSE, logical = TRUE)
registerDoMC(8)
getDoParWorkers()


##### Create separate CSVs

results_file = '~/github/bdr/pew-experiment/results/pew_simulation_results.csv'
nrows_file = system(paste0('wc -l ', results_file), intern = T)
nrows_file = as.numeric(strsplit(nrows_file, " ")[[1]][2])

chunk_size = 5259 * 12

fread(results_file, nrows = 10)

rows_processed = 0
while(rows_processed < nrows_file){
  temp = fread(results_file, nrows = chunk_size, skip = rows_processed + 1)
  setnames(temp, c("match_rate" , "n_bags", "n_landmarks", "refit_bags", "party", "model", "y_hat_dem", "y_hat_rep", "y_hat_oth" ))
  
  temp[, results_id := paste0('match', round(match_rate*100)
                , '_bags', n_bags
                , '_lmks', n_landmarks
                , '_refitbags', refit_bags
                , '_party',party)]
  
  results_ids = unique(temp$results_id)
  
  lapply(results_ids, function(id){
    file = paste0('~/github/bdr/pew-experiment/results/simulation/results_', id, '.csv')
    exists = file.exists(file)
    
    write.table(temp[results_id == id,], file = file, append = exists, col.names = !exists, row.names = FALSE, sep = ',')
  })
  

  
  rows_processed = rows_processed + chunk_size
}




pew_data = fread('data/data_recoded.csv')
results_summary = list()
file_dir = '~/github/bdr/pew-experiment/results/simulation/'
file_list = list.files(file_dir)

results_summary = foreach(i=1:length(file_list)) %dopar%{
  f = file_list[i]
  #f = 'results_match70_bags75_lmks200_refitbagsTRUE_partyonfile.csv'
  temp = fread(paste0(file_dir, '/', f))
  
  mses = rbindlist(lapply(unique(temp$model), function(m){
    data.table(results_id = temp$results_id[1]
               , model = m
               , mse = calcMSE(Y = as.numeric(unlist(pew_data[, c(y_dem, y_rep, y_oth)]))
            , Y_pred = as.numeric(unlist(temp[model == m, .(y_hat_dem, y_hat_rep, y_hat_oth)]))))
  }))
  
  mses = mses[, mse_rel := mse/mses$mse[model == 'logit_alldata']]
  
  cbind(temp[1,1:5], mses)
  
  
}

results_all = rbindlist(results_summary)



results_all[model == 'dr_cust' & mse_rel < 0.9 , results_id]

results_all[(results_all[model == 'dr_cust' & mse_rel < 0.9 , results_id][1] == results_id) & model == 'logit_alldata',]


which(results_all[,model == 'dr_sepbags' & mse_rel > 1,])

sim_params[104,]

ggplot(results_all, aes(x = n_bags, y = n_landmarks, color = mse)) + facet_grid(match_rate ~ model) + geom_point()

ggplot(results_all, aes(x = n_bags, y = mse_rel, color = model)) + geom_point()

ggplot(results_all[party == 'onfile' & model %in% c('dr_cust')]) + geom_density(aes(x = mse, color = factor(interaction(n_bags,match_rate)), lty = refit_bags))
ggplot(results_all[party == 'onfile' & model %in% c('dr')]) + geom_density(aes(x = mse, color = factor(interaction(n_bags,match_rate)), lty = refit_bags))



