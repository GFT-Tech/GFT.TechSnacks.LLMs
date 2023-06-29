using Azure;
using Azure.AI.OpenAI;
using GFT.Bot.Api.Models;

namespace GFT.Bot.Api.Services;

public class OpenAiService
{
    private readonly ILogger<OpenAiService> _logger;
    private readonly IConfiguration _configuration;
    public OpenAiService(ILogger<OpenAiService> logger, IConfiguration configuration)
    {
        _logger = logger;
        _configuration = configuration;

    }
    
    string system_role=@"You are a GFT Tech Scnaks Bot. Follow there instructions:
- Always salute the user and thank him to come to Tech Snacks
- Finish and sentences with 'And Life is good!'
- Whenever possible use Bullet Points to break your answer
";
    public async Task<ChatChoice> Prompt(PromptDto dto)
    {
        OpenAIClient client = new OpenAIClient(
            new Uri(_configuration["Azure-OpenAI-Uri"]),
            new AzureKeyCredential(_configuration["Azure-OpenAI-Key"]));
        
        Response<ChatCompletions> responseWithoutStream = await client.GetChatCompletionsAsync(
            dto.DeploymentOrModelName,
            new ChatCompletionsOptions()
            {
                Messages =
                {
                    new ChatMessage(ChatRole.System, system_role ),
                    new ChatMessage(ChatRole.User, dto.Question)
                },
                Temperature = (float)0.7,
                MaxTokens = 800,
                NucleusSamplingFactor = (float)0.95,
                FrequencyPenalty = 0,
                PresencePenalty = 0,
            });

        return responseWithoutStream.Value.Choices.FirstOrDefault();
    }
}