import { GetObjectCommand, PutObjectCommand, S3Client } from "@aws-sdk/client-s3";
import { getSignedUrl } from "@aws-sdk/s3-request-presigner";
import type { Env } from "../types";

const R2_BUCKET_NAME = "qwen-tts-studio";

const createR2Client = (env: Env): { client: S3Client; bucketName: string } => {
  const client = new S3Client({
    region: "auto",
    endpoint: env.R2_ENDPOINT_URL,
    credentials: {
      accessKeyId: env.R2_ACCESS_KEY_ID,
      secretAccessKey: env.R2_SECRET_ACCESS_KEY,
    },
  });

  return { client, bucketName: R2_BUCKET_NAME };
};

export const generatePresignedGet = async (
  env: Env,
  key: string,
  expiresIn = 3600
): Promise<string> => {
  const { client, bucketName } = createR2Client(env);
  const command = new GetObjectCommand({
    Bucket: bucketName,
    Key: key,
  });

  return getSignedUrl(client, command, { expiresIn });
};

export const generatePresignedPut = async (
  env: Env,
  key: string,
  contentType: string,
  expiresIn = 3600
): Promise<string> => {
  const { client, bucketName } = createR2Client(env);
  const command = new PutObjectCommand({
    Bucket: bucketName,
    Key: key,
    ContentType: contentType,
  });

  return getSignedUrl(client, command, { expiresIn });
};
