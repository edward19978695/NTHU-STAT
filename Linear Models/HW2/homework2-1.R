data = read.table("wrinkle.txt", header = T)

# a.
fit = lm(press ~ HCHO + catalyst + temp + time, data = data)
summary(fit)
names(summary(fit))

summary(fit)$r.squared


# b.
res = summary(fit)$residuals
res[res == max(res)]

# c.
mean(res)
median(res)


# d.
fitted_value = fit$fitted.values
cor(res, fitted_value)


# e.
cor(res, data$HCHO)


# f.
fit$coefficients[4] * 10


# e.
fit2 = lm(press ~ HCHO + catalyst + temp + time + (HCHO-catalyst), data = data)
summary(fit2)

fit3 = lm(press ~ HCHO + catalyst + temp + time + (HCHO/catalyst), data = data)
summary(fit3)













