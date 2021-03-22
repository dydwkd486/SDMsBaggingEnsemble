# load the library
library(biomod2)
# load our species data
for(cnt in 1:5){
  name_train=paste("external/species/train1vs",cnt,".csv",sep="")
  name_train
  DataSpecies <- read.csv(system.file(name_train,package="biomod2"))
  head(DataSpecies)
  
  # the name of studied species 연구된 종의 이름
  myRespName <- 'present.pseudo_absent'
  
  # the presence/absences data for our species 우리의 종에 대한 유무 데이터
  myResp <- as.numeric(DataSpecies[,myRespName])
  
  # the XY coordinates of species data 종 데이터의 xy좌표 
  myRespXY <- DataSpecies[c("dLon","dLat")]
  
  # 테스트
  name_test=paste("external/species/test1vs",cnt,".csv",sep="")
  DatatestSpecies <- read.csv(system.file(name_test,package="biomod2"))
  myevalRespName <- 'present.pseudo_absent'
  myevalResp<-as.numeric(DatatestSpecies[,myevalRespName])
  myevalRespXY <- DatatestSpecies[c("dLon","dLat")]
  
  # load the environmental raster layers (could be .img, ArcGIS
  # rasters or any supported format by the raster package)
  # Environmental variables extracted from Worldclim (bio_3, bio_4,
  # bio_7, bio_11 & bio_12)
  
  myExpl = stack( system.file( "external/bioclim/current/detail_03.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/detail_04.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/detail_05.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/detail_012.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/detail_013.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/detail_014.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/new/330.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/new/420.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/new/520.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/new/720.grd",
                               package="biomod2"),
                  system.file( "external/bioclim/current/Mosaic_cropland1.grd", #20
                               package="biomod2"),
                  system.file( "external/bioclim/current/Mosaic_vegetation1.grd", #30
                               package="biomod2"),
                  system.file( "external/bioclim/current/broadleaved_deciduous_forest1.grd", #50
                               package="biomod2"),
                  system.file( "external/bioclim/current/needleleaved_evergreen_forest1.grd", #70
                               package="biomod2"),
                  system.file( "external/bioclim/current/needleleaved_deciduous_or_evergreen_forest1.grd", #90
                               package="biomod2"),
                  system.file( "external/bioclim/current/Closed_to_open_mixed_broadleaved_and_needleleaved_forest1.grd", #100
                               package="biomod2"),
                  system.file( "external/bioclim/current/Mosaic_grassland1.grd", #120
                               package="biomod2"),
                  system.file( "external/bioclim/current/herbaceous_vegetation1.grd", #140
                               package="biomod2"),
                  system.file( "external/bioclim/current/Artificial_surfaces_and_associated_areas1.grd", #190
                               package="biomod2"),
                  system.file( "external/bioclim/current/Sparse_vegetation.grd", #150
                               package="biomod2"),
                  system.file( "external/bioclim/current/Bare_areas1.grd", #200
                               package="biomod2")
  )
  
  
  myBiomodData <- BIOMOD_FormatingData(resp.var = myResp,
                                       expl.var = myExpl,
                                       resp.xy = myRespXY,
                                       resp.name = myRespName,
                                       eval.resp.var = myevalResp,
                                       eval.expl.var = myExpl,
                                       eval.resp.xy =myevalRespXY 
  )
  
  myBiomodOption <- BIOMOD_ModelingOptions()
  # 3. Computing the models
  myBiomodModelOut <- BIOMOD_Modeling(
    myBiomodData,
    models = c("GLM","GBM", "CTA", "ANN", "FDA","MARS", "RF","SRE"),
    models.options = myBiomodOption,
    NbRunEval=5,
    DataSplit=80,
    VarImport=0,
    models.eval.meth = c('KAPPA','TSS','ROC'),
    SaveObj = TRUE,
    rescal.all.models = TRUE,
    do.full.models = FALSE,
    modeling.id = paste("train_testvs",cnt,"FirstModeling",sep=""))
  
  myBiomodModelEval <- get_evaluations(myBiomodModelOut)
  myBiomodModelEval
  
  
  # print the dimnames of this object
  dimnames(myBiomodModelEval)
  # let's print the TSS scores of Random Forest
  myBiomodModelEval["TSS","Testing.data","RF",,]
  
  # let's print the ROC scores of all selected models
  myBiomodModelEval["ROC","Testing.data",,,]
  
  # print variable importances
  get_variables_importance(myBiomodModelOut)
  
  
  # projection over the globe under current conditions
  myBiomodProj <- BIOMOD_Projection(
    modeling.output = myBiomodModelOut,
    new.env = myExpl,
    proj.name = paste('train_testvs',cnt,sep=""),
    selected.models = 'all',
    binary.meth = 'TSS',
    compress = 'xz',
    clamping.mask = F,
    output.format = '.grd')

  all_proj <- get_predictions(myBiomodProj)
  all_proj1 <- get_predictions(myBiomodModelOut)
  for(i in 1:40){
    sel_proj <- subset(all_proj,i)
    
    ## extract obs sites from projection
    extr_proj <- extract(sel_proj, myevalRespXY)
    
    ## test some threshold ( 0 - 1000 scaled ) to evaluate models quality
    thresh <- seq(0,1000,100)
    eval <- NULL
    for(th in thresh){
      eval <- rbind( eval, Find.Optim.Stat(Stat = 'TSS',
                                           Fit = extr_proj,
                                           Obs = myevalResp,
                                           Fixed.thresh = th) )
    }
    eval
    eval1<-as.data.frame(eval)
    writexl::write_xlsx(eval1, path = paste("present.pseudo.absent/proj_train_testvs",cnt,"/TSS_",myBiomodProj@models.projected[i],"eval.xlsx",sep=""))
  }
  
  for(i in 1:40){
    sel_proj <- subset(all_proj,i)
    
    ## extract obs sites from projection
    extr_proj <- extract(sel_proj, myevalRespXY)
    
    ## test some threshold ( 0 - 1000 scaled ) to evaluate models quality
    thresh <- seq(0,1000,100)
    eval <- NULL
    for(th in thresh){
      eval <- rbind( eval, Find.Optim.Stat(Stat = 'KAPPA',
                                           Fit = extr_proj,
                                           Obs = myevalResp,
                                           Fixed.thresh = th) )
    }
    eval
    eval1<-as.data.frame(eval)
    writexl::write_xlsx(eval1, path = paste("present.pseudo.absent/proj_train_testvs",cnt,"/KAPPA_",myBiomodProj@models.projected[i],"eval.xlsx",sep=""))
  }
  
  savexlsx<-as.data.frame(myBiomodModelEval)
  writexl::write_xlsx(savexlsx, path = paste("present.pseudo.absent/proj_train_testvs",cnt,"/savexlsx.xlsx",sep=""))
  
  rm(list=ls())
}