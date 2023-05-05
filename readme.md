# GFT Bot API

This project is simple as it gets. Follow these Steps to run on your own.

## Backend
The Backend is built using Dotnet. IF you do not have the SDK, go [here](https://dotnet.microsoft.com/en-us/download).
### Set Secrets (Azure-OpenAI-Uri & Azure-OpenAI-Key)
To run the demo, you need to create a dotnet secrets named `Azure-OpenAI-Uri` and `Azure-OpenAI-Key` by running this command on the project(/backend/GFT.Bot.Api) folder:
```bash
dotnet user-secrets set "Azure-OpenAI-Uri" "https://your-end-point.openai.azure.com"
dotnet user-secrets set "Azure-OpenAI-Key" "your_secret_value"
```
**Note**: Get your key and Endpoint Uri from the azure portal under the 'Keys and Endpoint' section

### Run and Have fun
Just open the GFT.Bot.Api.sln on your Visual Studio or Rider and run it.
As soon as it runs you should see a swagger ui on the 
http://localhost:8181/Swagger

Calls from the frontend will be made here:
http://localhost:8181/chat

Here a curl sample if you want to try it directly:
```bash
curl --location 'http://localhost:8181/chat' \
--header 'Content-Type: application/json' \
--data '{
    "Question":"What is the meaning of life",
    "DeploymentOrModelName":"gpt-play"
}'
```

## Frontend
You need angular to run it. If you do not have angular, just go [here](https://angular.io/guide/setup-local). If you have no Angular, you might nmeed to install NodeJs as well... As Long as you follow the Angular Setup Guide you should be fine!

### Install dependencies
Go to the /frontend/gft-bot folder and run:
```
npm install
```
Wait patiently! :-)

### Run and Have fun!
Once the dependencies are installed, just run a ng serve to start
```
ng serve
```

A browser will be opened on the following address:
http://localhost:4200

