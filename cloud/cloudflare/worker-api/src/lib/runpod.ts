import type { Env } from "../types";

const RUNPOD_SERVERLESS_BASE_URL = "https://api.runpod.ai/v2";
const RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql";

type GraphQlResponse<T> = {
  data?: T;
  errors?: Array<{ message: string }>;
};

const callGraphQl = async <T>(env: Env, query: string, variables: Record<string, unknown>) => {
  const response = await fetch(RUNPOD_GRAPHQL_URL, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ query, variables }),
  });

  if (!response.ok) {
    throw new Error(`RunPod GraphQL request failed (${response.status})`);
  }

  const payload = (await response.json()) as GraphQlResponse<T>;
  if (payload.errors && payload.errors.length > 0) {
    throw new Error(payload.errors.map((error) => error.message).join("; "));
  }
  if (!payload.data) {
    throw new Error("RunPod GraphQL response had no data");
  }

  return payload.data;
};

export const invokeServerless = async (
  env: Env,
  endpointId: string,
  input: Record<string, unknown>
): Promise<Record<string, unknown>> => {
  const url = `${RUNPOD_SERVERLESS_BASE_URL}/${endpointId}/runsync`;
  const response = await fetch(url, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
    body: JSON.stringify({ input }),
  });

  if (!response.ok) {
    const bodyText = await response.text();
    throw new Error(`RunPod serverless invocation failed (${response.status}): ${bodyText}`);
  }

  return (await response.json()) as Record<string, unknown>;
};

export const createPod = async (
  env: Env,
  templateId: string,
  gpuTypeId: string,
  envVars: Array<{ key: string; value: string }>
): Promise<{ podId: string; desiredStatus: string }> => {
  const query = `
    mutation CreatePod($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
      }
    }
  `;

  const input: Record<string, unknown> = {
    templateId,
    env: envVars,
    cloudType: "ALL",
    gpuCount: 1,
    containerDiskInGb: 200,
    volumeInGb: 0,
  };
  if (gpuTypeId) {
    input.gpuTypeId = gpuTypeId;
  }

  const data = await callGraphQl<{
    podFindAndDeployOnDemand: { id: string; desiredStatus: string };
  }>(env, query, { input });

  return {
    podId: data.podFindAndDeployOnDemand.id,
    desiredStatus: data.podFindAndDeployOnDemand.desiredStatus,
  };
};

export const terminatePod = async (env: Env, podId: string): Promise<boolean> => {
  const query = `
    mutation TerminatePod($input: PodTerminateInput!) {
      podTerminate(input: $input)
    }
  `;

  const data = await callGraphQl<{ podTerminate: boolean }>(env, query, {
    input: { podId },
  });

  return Boolean(data.podTerminate);
};

export const getPodStatus = async (
  env: Env,
  podId: string
): Promise<{ id: string; desiredStatus: string; runtimeStatus: string } | null> => {
  const query = `
    query PodStatus($podId: String!) {
      pod(input: { podId: $podId }) {
        id
        desiredStatus
        runtime {
          uptimeInSeconds
        }
        lastStatusChange
      }
    }
  `;

  const data = await callGraphQl<{
    pod: {
      id: string;
      desiredStatus: string;
      runtime?: { uptimeInSeconds?: number };
    } | null;
  }>(env, query, { podId });

  if (!data.pod) {
    return null;
  }

  const runtimeStatus = data.pod.runtime?.uptimeInSeconds !== undefined ? "running" : "pending";
  return {
    id: data.pod.id,
    desiredStatus: data.pod.desiredStatus,
    runtimeStatus,
  };
};
