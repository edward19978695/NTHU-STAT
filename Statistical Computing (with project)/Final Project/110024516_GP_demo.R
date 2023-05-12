library(shiny)
library(fields)
library(GpGp)



cov_func = c("Exponential", "Matern 15", "Matern 25", "Matern 35", "Matern 45")


n = 50
x <- seq(0,1, length=n)
grid1 <- expand.grid(x,x)
newloc = as.matrix(grid1)


ui = fluidPage(
    # Application title
    titlePanel("Cov. Function Setting"),
    
    sidebarLayout(
        # Sidebar with a slider and selection inputs
        sidebarPanel(
            fileInput("file", "Upload training data :",
                      multiple = FALSE,
                      accept = c("text/csv",
                                 "text/comma-separated-values,text/plain",
                                 ".csv")),
            hr(),
            selectInput("cov", "Choose a cov. family:",
                        choices = cov_func),
            sliderInput("sig_square",
                        "Variance parameter :",
                        min = 0.01,  max = 10, value = 0.01, step = 0.01),
            sliderInput("theta",
                        "Range parameter :",
                        min = 0.02,  max = 1,  value = 0.01, step = 0.01), 
            sliderInput("tau_square",
                        "Nugget effect :",
                        min = 0,  max = 5,  value = 0, step = 0.1)
        ),
        
        # Show 2-dim prediction
        mainPanel(
            tabsetPanel(type = "tabs",
                        tabPanel("Plot", plotOutput("plot", width = 650, height = 650)),
                        tabPanel("Summary", verbatimTextOutput("summary")),
                        tabPanel("Table", downloadButton("downloadData", "Download .csv"), tableOutput("table"))
            )
        )
    )
)

server = function(input, output) {
    fit = reactive({
        data = read.csv(input$file$datapath)
        if (max(data[,2]) > 1) data[,2] = data[,2]/max(data[,2])
        if (max(data[,3]) > 1) data[,3] = data[,3]/max(data[,3])
        fit_model(y=data[,4], locs=data[, 2:3],
                  covfun_name="matern_isotropic", m_seq=c(10,30))
    })
    
    para = reactive({
        if (input$cov == "Exponential") {
            nu = 0.5 } else if (input$cov == "Matern 15") {
                nu = 1.5 } else if (input$cov == "Matern 25") {
                    nu = 2.5} else if (input$cov == "Matern 35") {
                        nu = 3.5 } else {
                            nu = 4.5}
        c(input$sig_square, input$theta, nu, input$tau_square)
    })
    
    y = reactive({
        predictions(fit(), locs_pred=newloc, X_pred=rep(1,n^2), covparms = para())
    })
    
    table = reactive({
        data = read.csv(input$file$datapath)
        if (max(data[,2]) > 1) newloc[,1] = newloc[,1]*max(data[,2])
        if (max(data[,3]) > 1) newloc[,2] = newloc[,2]*max(data[,3])
        data.frame(X1 = newloc[,1], X2 = newloc[,2], Y = y())
    })

    # Fill in the spot we created for a plot
    output$plot <- renderPlot({
        quilt.plot(newloc, y(), xlim=c(0,1), ylim=c(0,1), zlim=range(y()),
                   cex.lab=1.5, cex.axis=1.5, nx=n, ny=n, pty="s", asp = 1)
    })
    
    output$summary = renderPrint({
        summary(y())
    })
    
    output$table = renderTable({
        table()
    })
    
    
    output$downloadData <- downloadHandler(
        filename = function() {
            paste(input$cov, "(", input$sig_square, ",", input$theta, ",", input$tau_square, ")", ".csv", sep = "")
        },
        content = function(file) {
            write.csv(table(), file, row.names = FALSE)
        }
    )
}

shinyApp(ui = ui, server = server)