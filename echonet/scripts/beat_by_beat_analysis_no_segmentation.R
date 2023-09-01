library(ggplot2)
library(stringr)
library(plyr)
library(dplyr)
library(lubridate)
library(reshape2)
library(scales)
library(ggthemes)
library(Metrics)

# set results directory
resultsDir <- "../output"

# load output from echonet video function
data <- read.csv(file.path(resultsDir, "echonet_video_EDES", "test_predictions.csv"), header = FALSE)
str(data)


dataNoAugmentation <- data[data$V2 == 0,]
str(dataNoAugmentation)


dataGlobalAugmentation <- data %>% group_by(V1) %>% summarize(meanPrediction = mean(V3), sdPred = sd(V3))
str(dataGlobalAugmentation)


beatByBeat <- dataGlobalAugmentation
colnames(beatByBeat)[1] <- "FileName"
str(beatByBeat)

### For use, need to specify file directory
fileLocation <- "../../EchoNet-Dynamic"
ActualNumbers <- read.csv(file.path(fileLocation, "FileList.csv"), colClasses = c("FileName" = "character"))
ActualNumbers <- ActualNumbers[c(1,2)]
ActualNumbers["FileName"] <- lapply(ActualNumbers["FileName"], function(x) paste(x, ".avi", sep = ""))
str(ActualNumbers)


dataNoAugmentation <- merge(dataNoAugmentation, ActualNumbers, by.x = "V1", by.y = "FileName", all.x = TRUE)
dataNoAugmentation$AbsErr <- abs(dataNoAugmentation$V3 - dataNoAugmentation$EF)
str(dataNoAugmentation)

summary(abs(dataNoAugmentation$V3 - dataNoAugmentation$EF))
# Mean of 4.216

rmse(dataNoAugmentation$V3,dataNoAugmentation$EF) 
## 5.56

modelNoAugmentation <- lm(dataNoAugmentation$EF ~ dataNoAugmentation$V3)
summary(modelNoAugmentation)$r.squared
# 0.79475


beatByBeat <- merge(beatByBeat, ActualNumbers, by.x = "FileName", by.y = "FileName", all.x = TRUE)
summary(abs(beatByBeat$meanPrediction - beatByBeat$EF))
# Mean of 4.051697

rmse(beatByBeat$meanPrediction, beatByBeat$EF) 
# 5.325237

modelBeatByBeat <- lm(beatByBeat$EF ~ beatByBeat$meanPrediction)
summary(modelBeatByBeat)$r.squared
# 0.8093174


# beatByBeatAnalysis <- merge(sizeRelevantFrames, data, by.x = c("FileName", "Frame"), by.y = c("V1", "V2"))
# str(beatByBeatAnalysis)
# 
# 
# MAEdata <- data.frame(counter = 1:500)
# MAEdata$sample <- -9999
# MAEdata$error <- -9999
# 
# str(MAEdata)
# 
# for (i in 1:500){
# 
# 
# samplingBeat <-  sample_n(beatByBeatAnalysis %>% group_by(FileName), 1 + floor((i-1)/100), replace = TRUE) %>% 
#     group_by(FileName) %>% dplyr::summarize(meanPred = mean(V3))
# samplingBeat <- merge(samplingBeat, ActualNumbers, by.x = "FileName", by.y = "FileName", all.x = TRUE)
# samplingBeat$error <- abs(samplingBeat$meanPred - samplingBeat$EF)
# 
# MAEdata$sample[i] <-  1 + floor((i-1)/100)
# MAEdata$error[i] <- mean(samplingBeat$error)
# 
# 
# }
# 
# str(MAEdata)
# 
# beatBoxPlot <- ggplot(data = MAEdata) + geom_boxplot(aes(x = sample, y = error, group = sample), outlier.shape = NA
# ) + theme_classic() + theme(legend.position = "none", 
#                             axis.text.y = element_text( size=7)) + xlab("Number of Sampled Beats") + 
#                             ylab("Mean Absolute Error") + scale_fill_brewer(palette = "Set1", direction = -1) 
# 
# beatBoxPlot

