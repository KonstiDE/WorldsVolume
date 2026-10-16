library(terra)

setwd("/media/caipi/Toshi/volume/sites/dl")


site <- "phenix1"

dsm <- vect(paste0(site, "/", site, ".tif"))
hole <- vect(paste0(site, "/", "hole.gpkg"))