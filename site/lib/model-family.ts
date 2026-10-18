export function modelFamily(model: string) {
  if (model.startsWith("Claude")) return "anthropic";
  if (model.startsWith("Gemini")) return "google";
  if (model.startsWith("GPT")) return "openai";
  if (model.startsWith("Kimi")) return "moonshot";
  if (model.startsWith("DeepSeek")) return "deepseek";
  return "xai";
}
