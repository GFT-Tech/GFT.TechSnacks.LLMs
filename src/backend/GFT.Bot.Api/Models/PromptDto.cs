using Microsoft.AspNetCore.Mvc.Infrastructure;

namespace GFT.Bot.Api.Models;

public class PromptDto
{
    public string Question { get; set; }
    public string DeploymentOrModelName { get; set; }
}