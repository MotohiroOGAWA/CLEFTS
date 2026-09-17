export interface JobInfo {
  id: string; type: 'training' | 'preparation' | 'prediction'; name: string;
  status: 'running' | 'completed' | 'failed' | 'cancelled';
  pid?: number; startedAt: string; finishedAt?: string;
  outputDir: string; logPath: string; command: string[]; epochs?: number; detached?: boolean; resultsPath?: string;
  error?: string; exitCode?: number | null;
}
export type WorkbenchRequest =
  | { type: 'workbench/ready' | 'workbench/environment' | 'workbench/settings' | 'workbench/terminal' }
  | { type: 'workbench/stop' | 'workbench/logs' | 'workbench/openOutput' | 'workbench/predictionResult'; jobId: string }
  | { type: 'workbench/modelConfig'; path: string; open: boolean }
  | { type: 'workbench/saveModelConfig'; json: string };
export type WorkbenchEvent =
  | { type: 'workbench/jobs'; jobs: JobInfo[] }
  | { type: 'workbench/jobSelected'; jobId: string }
  | { type: 'workbench/navigate'; page: string }
  | { type: 'workbench/logs'; job: JobInfo; text: string }
  | { type: 'workbench/error'; error: string }
  | { type: 'workbench/modelConfig'; config: Record<string, unknown> }
  | { type: 'workbench/environment'; error?: string; data?: { torch: string; cuda: boolean; gpu: string | null; root: string; python: string } };

export type ParameterRequest =
  | { type: 'parameters/pick'; target: 'data' | 'training' }
  | { type: 'parameters/import'; target: 'data' | 'training'; name: string; json: string }
  | { type: 'parameters/load'; target: 'data' | 'training'; path: string };
export type ParameterEvent =
  | { type: 'parameters/loaded'; target: 'data' | 'training'; path: string; modelConfig: Record<string, unknown> }
  | { type: 'parameters/error'; target: 'data' | 'training'; error: string };

export type DataRequest =
  | { type: 'data/preview' | 'data/check'; target: 'train' | 'validation' | 'home'; requestId: string; path: string; mapping?: Record<string, string>; fragmenterParams?: Record<string, unknown> }
  | { type: 'data/preflight'; requestId: string; config: Record<string, unknown> }
  | { type: 'data/model'; requestId: string; path: string };
export type ResourceRequest =
  | { type: 'library/load' | 'shell/document' | 'shell/github' }
  | { type: 'library/copy' | 'library/open'; path: string }
  | { type: 'library/pick'; target: 'quickTrainDir' | 'quickValDir' | 'quickTrainOutput' | 'quickCheckpoint' | 'quickPredictModel' };

export type DefaultRequest = { type: 'defaults/save'; workflow: 'data' | 'training' | 'prediction'; config: Record<string, unknown> };
export type DefaultEvent = { type: 'defaults/saved' | 'defaults/error'; workflow: 'data' | 'training' | 'prediction'; error?: string };
