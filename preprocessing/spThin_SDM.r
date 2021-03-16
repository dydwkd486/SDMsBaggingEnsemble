library(spThin)

data_csv <- read.csv("R/win-library/3.6/biomod2/external/species/name.csv")

thinned_dataset_full <-
  thin( loc.data = data_csv, 
        lat.col = "decimalLatitude", long.col = "decimalLongitude",
        thin.par = 10, reps = 100, 
        locs.thinned.list.return = TRUE, 
        write.files = TRUE, 
        max.files = 1, 
        out.dir = "spthin/", out.base = "name",write.log.file = FALSE)