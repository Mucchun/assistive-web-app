// Vercel Edge Function — keeps the Anthropic API key server-side
export const config = { runtime: "edge" };

export default async function handler(req: Request): Promise<Response> {
  if (req.method !== "POST") {
    return new Response("Method not allowed", { status: 405 });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return new Response(JSON.stringify({ error: "API key not configured on server." }), {
      status: 500,
      headers: { "Content-Type": "application/json" },
    });
  }

  const { prompt, imageBase64 } = await req.json() as {
    prompt: string;
    imageBase64: string;
  };

  const upstream = await fetch("https://api.anthropic.com/v1/messages", {
    method: "POST",
    headers: {
      "x-api-key": apiKey,
      "anthropic-version": "2023-06-01",
      "content-type": "application/json",
    },
    body: JSON.stringify({
      model: "claude-haiku-4-5-20251001",
      max_tokens: 300,
      messages: [{
        role: "user",
        content: [
          { type: "image", source: { type: "base64", media_type: "image/jpeg", data: imageBase64 } },
          { type: "text", text: prompt },
        ],
      }],
    }),
  });

  const data = await upstream.json() as { content?: Array<{ type: string; text?: string }> };
  const text = data.content?.find(b => b.type === "text")?.text ?? "";

  return new Response(JSON.stringify({ text }), {
    headers: { "Content-Type": "application/json" },
  });
}
