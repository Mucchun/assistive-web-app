import type { VercelRequest, VercelResponse } from "@vercel/node";

export default async function handler(req: VercelRequest, res: VercelResponse) {
  if (req.method !== "POST") {
    return res.status(405).json({ error: "Method not allowed" });
  }

  const apiKey = process.env.ANTHROPIC_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "API key not configured on server." });
  }

  const { prompt, imageBase64 } = req.body as { prompt: string; imageBase64: string };
  if (!prompt || !imageBase64) {
    return res.status(400).json({ error: "Missing prompt or image." });
  }

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

  const data = await upstream.json() as {
    content?: Array<{ type: string; text?: string }>;
    error?: { type: string; message: string };
  };

  if (!upstream.ok || data.error) {
    const msg = data.error?.message ?? `Upstream error ${upstream.status}`;
    return res.status(upstream.ok ? 500 : upstream.status).json({ error: msg });
  }

  const text = data.content?.find(b => b.type === "text")?.text ?? "";
  return res.status(200).json({ text });
}
