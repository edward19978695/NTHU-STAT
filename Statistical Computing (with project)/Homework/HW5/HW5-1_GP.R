library(fields)
library(GpGp)
train_data = read.csv("train.csv")
test_data = read.csv("test.csv")
idx = sample(1:1000, 100)
quilt.plot(train_data[idx,2:3], train_data[idx,4], xlim=c(0,1), ylim=c(0,1), zlim=range(train_data[,4]), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
fit <- fit_model(y=train_data[idx,4], locs=train_data[idx, 2:3], 
                  covfun_name="matern25_isotropic", m_seq=c(10,30))
summary(fit)

n = 100
x <- seq(0,1, length=n)
grid1 <- expand.grid(x,x)


newloc = as.matrix(grid1)
colnames(newloc) = c("x1","x2")
y <- predictions(fit, locs_pred=newloc, X_pred=rep(1,10000)) 

y <- predictions(fit, locs_pred=newloc, X_pred=rep(1,10000), covparms = c(2,1/10,0))

par(mfrow = c(1,2))
quilt.plot(train_data[,2:3], train_data[,4], xlim=c(0,1), ylim=c(0,1), zlim=range(train_data[,4]), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Training data")
quilt.plot(newloc, y, xlim=c(0,1), ylim=c(0,1), zlim=range(y), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Testing data")


fit2 <- fit_model(y=train_data[,4], locs=train_data[, 2:3], 
                 covfun_name="matern25_isotropic", m_seq=c(10,30))
summary(fit2)
y2 <- predictions(fit2, locs_pred=newloc, X_pred=rep(1,20000))
par(mfrow = c(1,2))
quilt.plot(train_data[,2:3], train_data[,4], xlim=c(0,1), ylim=c(0,1), zlim=range(train_data[,4]), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Training data")
quilt.plot(newloc, y2, xlim=c(0,1), ylim=c(0,1), zlim=range(y2), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Testing data")


fit3 <- fit_model(y=train_data[,4], locs=train_data[, 2:3], 
                  covfun_name="matern15_scaledim", m_seq=c(10,30))
summary(fit3)
y3 <- predictions(fit3, locs_pred=newloc, X_pred=rep(1,20000))
par(mfrow = c(1,2))
quilt.plot(train_data[,2:3], train_data[,4], xlim=c(0,1), ylim=c(0,1), zlim=range(train_data[,4]), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Training data")
quilt.plot(newloc, y3, xlim=c(0,1), ylim=c(0,1), zlim=range(y3), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Testing data")


fit4 <- fit_model(y=train_data[,4], locs=train_data[, 2:3], 
                  covfun_name="exponential_isotropic", m_seq=c(10,30))
summary(fit4)
y4 <- predictions(fit4, locs_pred=newloc, X_pred=rep(1,20000))
par(mfrow = c(1,2))
quilt.plot(train_data[,2:3], train_data[,4], xlim=c(0,1), ylim=c(0,1), zlim=range(train_data[,4]), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Training data")
quilt.plot(newloc, y4, xlim=c(0,1), ylim=c(0,1), zlim=range(y3), 
           cex.lab=1.5, cex.axis=1.5, nx=100, ny=100, pty="s")
title("Testing data")


output = data.frame(test_id = test_data$test_id, 
                    y = y)
write.csv(output, "test2.csv",row.names = F)
