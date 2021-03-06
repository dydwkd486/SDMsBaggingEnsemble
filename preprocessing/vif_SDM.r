library(usdm)
r1 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/01.tif")
r2 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/02.tif")
r3 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/03.tif")
r4 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/04.tif")
r5 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/05.tif")
r6 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/06.tif")
r7 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/07.tif")
r8 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/08.tif")
r9 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/09.tif")
r10 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/010.tif")
r11 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/011.tif")
r12 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/012.tif")
r13 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/013.tif")
r14 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/014.tif")
r15 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/015.tif")
r16 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/016.tif")
r17 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/017.tif")
r18 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/018.tif")
r19 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/019.tif")
r110 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/110.tif")
r120 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/120.tif")
r130 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/130.tif")
r140 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/140.tif")
r150 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/150.tif")
r160 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/160.tif")
r170 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/170.tif")
r210 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/210.tif")
r220 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/220.tif")
r230 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/230.tif")
r240 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/240.tif")
r250 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/250.tif")
r310 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/310.tif")
r320 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/320.tif")
r330 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/330.tif")
r410 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/410.tif")
r420 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/420.tif")
r430 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/430.tif")
r510 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/510.tif")
r520 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/520.tif")
r610 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/610.tif")
r620 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/620.tif")
r720 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/scaled/720.tif")
rglob_20 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_20.tif")
rglob_30 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_30.tif")
rglob_50 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_50.tif")
rglob_70 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_70.tif")
rglob_90 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_90.tif")
rglob_100 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_100.tif")
rglob_110 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_110.tif")
rglob_120 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_120.tif")
rglob_140 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_140.tif")
rglob_150 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_150.tif")
rglob_190 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_190.tif")
rglob_200 <- raster("F:/github/bird/code/sdmdl/data/gis/layers/non-scaled/glob_200.tif")

testRaster <- raster::addLayer(r1,r2,r3,r4,r5,r6,r7,r8,r9,r10,r11,r12,r13,r14,r15,r16,r17,r18,r19,r110,r120,r130,r140,r150,r160,r170,r210,r220,r230,r240,r250,r310,r320,r330,r410,r420,r510,r520,r610,r620,r720,rglob_20,rglob_30,rglob_50,rglob_70,rglob_90,rglob_100,rglob_110,rglob_120,rglob_140,rglob_150,rglob_190,rglob_200)
r <- brick(testRaster) 
r 
vif(r) # calculates vif for the variables in r