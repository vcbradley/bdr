


# function fixed from LICORS package
kmeanspp = function(data, k = 2, start = "random", iter.max = 100, nstart = 10, only_centers = F,
                    ...){
  kk <- k
  if (length(dim(data)) == 0) {
    data <- matrix(data, ncol = 1)
  }else {
    data <- cbind(data)
  }
  num.samples <- nrow(data)
  ndim <- ncol(data)
  out <- list()
  out$tot.withinss <- Inf
  for (restart in seq_len(nstart)) {
    cat('Restart ', restart, '\n')
    center_ids <- rep(0, length = kk)
    
    cat('Getting start \n')
    if (start == "random") {
      center_ids[1:2] = sample.int(num.samples, 1)
    } else if (start == "normal") {
      data.avg <- colMeans(data)
      data.cov <- cov(data)
      center_ids[1:2] = which.min(dmvnorm(data, mean = data.avg, 
                                          sigma = data.cov))
    } else {
      center_ids[1:2] = start
    }
    cat('Calc first dist \n')
    # calc first dist
    dists <- dista(x = data, xnew = matrix(data[center_ids[1], ], nrow = 1, ncol = ncol(data)), square = TRUE)
    
    for (ii in 2:kk) {
      cat('Iter ', ii, '\n')
      cat('\tGetting min \n')
      probs <- apply(dists, 2, min)
  
      probs[center_ids] <- 0
      #center_ids[ii] <- sample.int(num.samples, 1, prob = probs)
      center_ids[ii] <- sample(c(1:num.samples)[probs > 0], prob = probs[probs > 0], size = 1)
      
      # calc new dists
      cat('\tCalc new dist \n')
      dists_new <- dista(x = data, xnew = matrix(data[center_ids[ii], ], nrow = 1, ncol = ncol(data)), square = TRUE)
      dists = rbind(dists, dists_new)
    }
    #cat(paste('n unique: ', nrow(unique(data[center_ids, ]))))
    #cat('\n')
    
    if(!only_centers){
      tmp.out <- kmeans(data, centers = data[center_ids, ], 
                        iter.max = iter.max, ...)
      tmp.out$inicial.centers <- data[center_ids, ]
      tmp.out$center_ids <- center_ids
      if (tmp.out$tot.withinss < out$tot.withinss) {
        out <- tmp.out
      }
    }else{
      out = center_ids
    }
    
  }
  invisible(out)
}


#### Code implementing kmeans++ initialization manually/not from LICORS package
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
