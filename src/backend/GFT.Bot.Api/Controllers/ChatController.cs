using Azure.AI.OpenAI;
using GFT.Bot.Api.Models;
using GFT.Bot.Api.Services;
using Microsoft.AspNetCore.Mvc;

namespace GFT.Bot.Api.Controllers;

[ApiController]
[Route("[controller]")]
public class ChatController : ControllerBase
{
    private readonly ILogger<ChatController> _logger;
    private readonly OpenAiService _openAiService;
    public ChatController(ILogger<ChatController> logger, OpenAiService openAiService)
    {
        _logger = logger;
        _openAiService = openAiService;
    }

    
    [HttpPost(Name = "Prompt")]
    public async Task<ActionResult<ChatMessage>> PromptAsync([FromBody]PromptDto prompt)
    {
        var chatResponse = await _openAiService.Prompt(prompt);
        return Ok(chatResponse.Message);
    }
}