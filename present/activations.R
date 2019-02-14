# Taking a look at activation functions

library(dplyr)
library(ggplot2)

softplus <- function (x) log(exp(x) + 1)
softmax <- function(x) exp(x)/sum(exp(x))

plot(softmax(-10:5))
lines(softplus(-10:5))


rng <- -10:5

sm <- tibble(rng, softmax(rng), "SoftMax")
sp <- tibble(rng, softplus(rng), "SoftPlus")
cn <- c("Range", "Activation Value", "Activation Type")

colnames(sm) = cn
colnames(sp) = cn

activations <- bind_rows(sm,sp)

f1 <- ggplot(activations, aes(`Activation Type`, `Activation Value`)) +
    geom_boxplot() +
    labs(
        x = "Activation Type", y="Activation Values",
        title="Range of values when x = range(-5:10)") +
    theme(
        axis.text.x = element_text(size=14),
        axis.text.y = element_text(size=14)
    )

ggsave("imgs/activations.png", f1)
