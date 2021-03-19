library(boot)
data_csv <- read.csv('Hynobius_leechii_total_env_dataframe_train.csv')
set.seed(20)


result <- sample(1:10,                   
                          size = nrow(data_csv),   
                          replace = T,          
                          prob = c(0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1))  

library(boot)
samplemedian <- function(d, i) {
	return(mean(d[i]))
}

b = boot(data_csv[result ==1,]$present.pseudo_absent, samplemedian, R=1000)

write.csv(data_csv[result ==1,],'train1.csv')