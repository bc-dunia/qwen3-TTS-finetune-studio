import type { Env } from "../types";

const RUNPOD_SERVERLESS_BASE_URL = "https://api.runpod.ai/v2";
const RUNPOD_GRAPHQL_URL = "https://api.runpod.io/graphql";
const RUNPOD_REST_BASE_URL = "https://rest.runpod.io/v1";

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

const callRest = async <T>(env: Env, path: string): Promise<T> => {
  const response = await fetch(`${RUNPOD_REST_BASE_URL}${path}`, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
  });

  if (!response.ok) {
    throw new Error(`RunPod REST request failed (${response.status}): ${await response.text()}`);
  }

  return (await response.json()) as T;
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

export const invokeServerlessAsync = async (
  env: Env,
  endpointId: string,
  input: Record<string, unknown>
): Promise<Record<string, unknown>> => {
  const url = `${RUNPOD_SERVERLESS_BASE_URL}/${endpointId}/run`;
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
    throw new Error(`RunPod async invocation failed (${response.status}): ${bodyText}`);
  }

  return (await response.json()) as Record<string, unknown>;
};

export const getServerlessStatus = async (
  env: Env,
  endpointId: string,
  runId: string
): Promise<Record<string, unknown>> => {
  const url = `${RUNPOD_SERVERLESS_BASE_URL}/${endpointId}/status/${runId}`;
  const response = await fetch(url, {
    method: "GET",
    headers: {
      Authorization: `Bearer ${env.RUNPOD_API_KEY}`,
    },
  });

  if (!response.ok) {
    const bodyText = await response.text();
    throw new Error(`RunPod status request failed (${response.status}): ${bodyText}`);
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

export const createPodDirect = async (
  env: Env,
  input: {
    gpuTypeId: string;
    envVars: Array<{ key: string; value: string }>;
    imageName: string;
    dockerArgs?: string | null;
    name: string;
    cloudType?: "ALL" | "COMMUNITY" | "SECURE";
    containerRegistryAuthId?: string | null;
    ports?: string[] | null;
    volumeMountPath?: string | null;
  }
): Promise<{ podId: string; desiredStatus: string }> => {
  const query = `
    mutation CreatePodDirect($input: PodFindAndDeployOnDemandInput!) {
      podFindAndDeployOnDemand(input: $input) {
        id
        desiredStatus
      }
    }
  `;

  const payload: Record<string, unknown> = {
    cloudType: input.cloudType ?? "ALL",
    gpuCount: 1,
    gpuTypeId: input.gpuTypeId,
    containerDiskInGb: 200,
    volumeInGb: 0,
    imageName: input.imageName,
    name: input.name,
    env: input.envVars,
  };
  if (input.dockerArgs && input.dockerArgs.trim()) {
    payload.dockerArgs = input.dockerArgs.trim();
  }
  if (input.containerRegistryAuthId) {
    payload.containerRegistryAuthId = input.containerRegistryAuthId;
  }
  if (input.ports && input.ports.length > 0) {
    payload.ports = input.ports.join(",");
  }
  if (input.volumeMountPath) {
    payload.volumeMountPath = input.volumeMountPath;
  }

  const data = await callGraphQl<{
    podFindAndDeployOnDemand: { id: string; desiredStatus: string };
  }>(env, query, { input: payload });

  return {
    podId: data.podFindAndDeployOnDemand.id,
    desiredStatus: data.podFindAndDeployOnDemand.desiredStatus,
  };
};

export const getTemplateById = async (
  env: Env,
  templateId: string
): Promise<{
  id: string;
  imageName?: string | null;
  containerRegistryAuthId?: string | null;
  ports?: string[] | null;
  volumeMountPath?: string | null;
  dockerEntrypoint?: string[] | null;
  dockerStartCmd?: string[] | null;
  isServerless?: boolean;
} | null> => {
  try {
    return await callRest<{
      id: string;
      imageName?: string | null;
      containerRegistryAuthId?: string | null;
      ports?: string[] | null;
      volumeMountPath?: string | null;
      dockerEntrypoint?: string[] | null;
      dockerStartCmd?: string[] | null;
      isServerless?: boolean;
    }>(env, `/templates/${templateId}`);
  } catch (error) {
    if (error instanceof Error && error.message.includes("(404)")) {
      return null;
    }
    throw error;
  }
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
): Promise<{
  id: string;
  desiredStatus: string;
  runtimeStatus: string;
  imageName: string | null;
  dockerArgs: string | null;
  templateId: string | null;
  machineType: string | null;
  uptimeSeconds: number | null;
  createdAt: string | null;
  lastStartedAt: string | null;
  lastStatusChange: string | null;
  runtime?: {
    uptimeInSeconds?: number;
    container?: {
      cpuPercent?: number;
      memoryPercent?: number;
    };
    gpus?: Array<{
      id?: string;
      gpuUtilPercent?: number;
      memoryUtilPercent?: number;
    }>;
  };
  latestTelemetry?: {
    state?: string;
    time?: string;
    cpuUtilization?: number;
    memoryUtilization?: number;
    lastStateTransitionTimestamp?: number;
  };
  machine?: {
    podHostId?: string;
    secureCloud?: boolean;
    supportPublicIp?: boolean;
    note?: string | null;
  };
} | null> => {
  const query = `
    query PodStatus($podId: String!) {
      pod(input: { podId: $podId }) {
        id
        desiredStatus
        imageName
        dockerArgs
        templateId
        machineType
        uptimeSeconds
        createdAt
        lastStartedAt
        lastStatusChange
        runtime {
          uptimeInSeconds
          container {
            cpuPercent
            memoryPercent
          }
          gpus {
            id
            gpuUtilPercent
            memoryUtilPercent
          }
        }
        latestTelemetry {
          state
          time
          cpuUtilization
          memoryUtilization
          lastStateTransitionTimestamp
        }
        machine {
          podHostId
          secureCloud
          supportPublicIp
          note
        }
      }
    }
  `;

  const data = await callGraphQl<{
    pod: {
      id: string;
      desiredStatus: string;
      imageName?: string | null;
      dockerArgs?: string | null;
      templateId?: string | null;
      machineType?: string | null;
      uptimeSeconds?: number | null;
      createdAt?: string | null;
      lastStartedAt?: string | null;
      lastStatusChange?: string | null;
      runtime?: { uptimeInSeconds?: number };
      latestTelemetry?: {
        state?: string;
        time?: string;
        cpuUtilization?: number;
        memoryUtilization?: number;
        lastStateTransitionTimestamp?: number;
      };
      machine?: {
        podHostId?: string;
        secureCloud?: boolean;
        supportPublicIp?: boolean;
        note?: string | null;
      };
    } | null;
  }>(env, query, { podId });

  if (!data.pod) {
    return null;
  }

  const telemetryState = String(data.pod.latestTelemetry?.state ?? "").trim().toLowerCase();
  const runtimeStatus = telemetryState || (
    data.pod.runtime?.uptimeInSeconds !== undefined
      ? (data.pod.runtime.uptimeInSeconds > 0 ? "running" : "created")
      : "pending"
  );
  return {
    id: data.pod.id,
    desiredStatus: data.pod.desiredStatus,
    runtimeStatus,
    imageName: data.pod.imageName ?? null,
    dockerArgs: data.pod.dockerArgs ?? null,
    templateId: data.pod.templateId ?? null,
    machineType: data.pod.machineType ?? null,
    uptimeSeconds: data.pod.uptimeSeconds ?? null,
    createdAt: data.pod.createdAt ?? null,
    lastStartedAt: data.pod.lastStartedAt ?? null,
    lastStatusChange: data.pod.lastStatusChange ?? null,
    runtime: data.pod.runtime,
    latestTelemetry: data.pod.latestTelemetry,
    machine: data.pod.machine,
  };
};
