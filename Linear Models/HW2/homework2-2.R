# a.
df = read.table("swiss.txt", header = T)
View(df)
fit = lm(Fertility ~ Agriculture + Examination + Education + 
             Catholic + Mortality, data = df)
summary(fit)
summary(fit)$cov # (X^T X)^-1
summary(fit)$sigma^2
summary(fit)$cov * (summary(fit)$sigma^2) # cov matrix of beta hat


# b.
res = fit$residuals
fit2 = lm(res ~ Agriculture + Examination + Education + 
                 Catholic + Mortality, data = df)
summary(fit2)


# c.
fitted_vl = fit$fitted.values
fit3 = lm(fitted_vl ~ Agriculture + Examination + Education + 
              Catholic + Mortality, data = df)
summary(fit3)



