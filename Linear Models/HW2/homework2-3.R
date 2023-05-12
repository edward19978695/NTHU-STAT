data = read.table("E2.9.txt", skip = 2)
names(data) = c("year", "k_20", "k_36", "k_37", "l_20", "l_36", "l_37", "v_20", "v_36", "v_37")


# For food and kindred products (20)
# a.
fit1_20 = lm(log(v_20) ~ log(k_20) + log(l_20), data = data)
summary(fit1_20)$coef

# b.
fit2_20 = lm(log(v_20) ~ log(k_20/l_20), offset = log(l_20), data = data)
summary(fit2_20)$coef

# c.
fit3_20 = lm(log(v_20) ~ log(k_20) + log(l_20) + year, data = data)
summary(fit3_20)$coef

# d.
fit4_20 = lm(log(v_20) ~ log(k_20/l_20) + year, offset = log(l_20), data = data)
summary(fit4_20)$coef




# For (36)
# a.
fit1_36 = lm(log(v_36) ~ log(k_36) + log(l_36), data = data)
summary(fit1_36)$coef

# b.
fit2_36 = lm(log(v_36) ~ log(k_36/l_36), offset = log(l_36), data = data)
summary(fit2_36)$coef

# c.
fit3_36 = lm(log(v_36) ~ log(k_36) + log(l_36) + year, data = data)
summary(fit3_36)$coef

# d.
fit4_36 = lm(log(v_36) ~ log(k_36/l_36) + year, offset = log(l_36), data = data)
summary(fit4_36)$coef



# For (37)
# a.
fit1_37 = lm(log(v_37) ~ log(k_37) + log(l_37), data = data)
summary(fit1_37)$coef

# b.
fit2_37 = lm(log(v_37) ~ log(k_37/l_37), offset = log(l_37), data = data)
summary(fit2_37)$coef

# c.
fit3_37 = lm(log(v_37) ~ log(k_37) + log(l_37) + year, data = data)
summary(fit3_37)$coef

# d.
fit4_37 = lm(log(v_37) ~ log(k_37/l_37) + year, offset = log(l_37), data = data)
summary(fit4_37)$coef
