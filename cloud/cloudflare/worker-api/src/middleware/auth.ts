import { createMiddleware } from "hono/factory";
import type { AppContext } from "../types";

const unauthorizedResponse = () =>
  Response.json(
    {
      detail: {
        message: "Invalid API key",
      },
    },
    { status: 401 }
  );

const isTruthyEnvValue = (value: string | undefined): boolean => {
  if (!value) {
    return false;
  }
  return ["1", "true", "yes", "on"].includes(value.trim().toLowerCase());
};

const extractBearerToken = (authorizationHeader: string | undefined): string | null => {
  if (!authorizationHeader) {
    return null;
  }
  const spaceIndex = authorizationHeader.indexOf(" ");
  if (spaceIndex === -1) {
    return null;
  }
  const scheme = authorizationHeader.slice(0, spaceIndex);
  const token = authorizationHeader.slice(spaceIndex + 1).trim();
  if (scheme.toLowerCase() !== "bearer" || !token) {
    return null;
  }
  return token;
};

export const authMiddleware = createMiddleware<AppContext>(async (c, next) => {
  if (c.req.path === "/") {
    await next();
    return;
  }

  if (isTruthyEnvValue(c.env.ALLOW_ANONYMOUS_ACCESS) || !c.env.API_KEY?.trim()) {
    await next();
    return;
  }

  const xiApiKey = c.req.header("xi-api-key");
  const bearerToken = extractBearerToken(c.req.header("authorization"));
  const providedKey = xiApiKey ?? bearerToken;

  if (!providedKey || providedKey !== c.env.API_KEY) {
    return unauthorizedResponse();
  }

  await next();
});
