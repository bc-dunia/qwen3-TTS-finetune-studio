import { Hono } from "hono";
import { authMiddleware } from "../middleware/auth";
import type { AppContext, Model } from "../types";

type ModelRow = {
  model_id: string;
  name: string;
  description: string | null;
  can_do_text_to_speech: number;
  can_be_finetuned: number;
  max_characters_request: number;
  languages_json: string;
};

const app = new Hono<AppContext>();
app.use("*", authMiddleware);

app.get("/", async (c) => {
  const result = await c.env.DB.prepare("SELECT * FROM models ORDER BY name ASC").all<ModelRow>();
  const models: Model[] = (result.results ?? []).map((row) => ({
    model_id: row.model_id,
    name: row.name,
    description: row.description,
    can_do_text_to_speech: row.can_do_text_to_speech === 1,
    can_be_finetuned: row.can_be_finetuned === 1,
    max_characters_request: row.max_characters_request,
    languages: (() => {
      try {
        return JSON.parse(row.languages_json) as Array<{ language_id: string; name: string }>;
      } catch {
        return [];
      }
    })(),
  }));

  return c.json({ models });
});

export default app;
