data1 = read.table("uswagesall.txt", header = T)
summary(data1)
data1$exper = abs(data1$exper)

# 整理資料

data1$race = factor(data1$race)
levels(data1$race) = c("White", "Black")
data1$smsa = factor(data1$smsa)
levels(data1$smsa) = c("Other", "in SMSA")
data1$ne = factor(data1$ne)
levels(data1$ne) = c("Other", "in NE")
data1$mw = factor(data1$mw)
levels(data1$mw) = c("Other", "in Mw")
data1$so = factor(data1$so)
levels(data1$so) = c("Other", "in So")
data1$we = factor(data1$we)
levels(data1$we) = c("Other", "in We")
data1$pt = factor(data1$pt)
levels(data1$pt) = c("No", "Part Time")
summary(data1)
head(sort(data1$wage, decreasing = T)) # 發現極富

# 對財富畫圖
par(mfrow = c(1,2))
plot(density(data1$wage, na.rm = T))
plot(sort(data1$wage), pch = ".")

# 去除極端值
Q1 = 308.64
Q3 = 783.48
IQR = Q3 - Q1
logic = Q1-1.5*IQR<data1$wage & data1$wage<Q3+1.5*IQR
new_wage = data1$wage[logic]
par(mfrow = c(1,3))
hist(new_wage)
plot(density(new_wage, na.rm = T))
plot(sort(new_wage), pch = ".")

data1 = data1[logic, ]
summary(data1)


# 做回歸
gfit = lm(wage ~ educ+exper+race+smsa+ne+mw+so+we+pt, 
          data = data1)
summary(gfit)


# 兩兩做圖
par(mfrow = c(1,2))
plot(wage ~ educ, data1)
plot(wage ~ exper, data1, pch = ".")


par(mfrow = c(1,4))
plot(wage ~ pt, data1)
plot(wage ~ race, data1)
plot(wage ~ smsa, data1)
plot(wage ~ so, data1)
