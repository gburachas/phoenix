/**
 * @generated SignedSource<<bb44636996ebc1b9022f8a67f3d19fa4>>
 * @lightSyntaxTransform
 * @nogrep
 */

/* tslint:disable */
/* eslint-disable */
// @ts-nocheck

import { ConcreteRequest } from 'relay-runtime';
export type DataGenerationPageQuery$variables = Record<PropertyKey, never>;
export type DataGenerationPageQuery$data = {
  readonly dataGenerationJobs: {
    readonly edges: ReadonlyArray<{
      readonly node: {
        readonly completedAt: string | null;
        readonly corpusSource: string;
        readonly createdAt: string;
        readonly id: string;
        readonly name: string;
        readonly status: string;
      };
    }>;
  };
  readonly llmAdapters: {
    readonly edges: ReadonlyArray<{
      readonly node: {
        readonly canGenerate: boolean;
        readonly canJudge: boolean;
        readonly createdAt: string;
        readonly id: string;
        readonly modelName: string;
        readonly name: string;
        readonly provider: string;
      };
    }>;
  };
};
export type DataGenerationPageQuery = {
  response: DataGenerationPageQuery$data;
  variables: DataGenerationPageQuery$variables;
};

const node: ConcreteRequest = (function(){
var v0 = [
  {
    "kind": "Literal",
    "name": "first",
    "value": 100
  }
],
v1 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "id",
  "storageKey": null
},
v2 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "name",
  "storageKey": null
},
v3 = {
  "alias": null,
  "args": null,
  "kind": "ScalarField",
  "name": "createdAt",
  "storageKey": null
},
v4 = [
  {
    "alias": null,
    "args": (v0/*: any*/),
    "concreteType": "DataGenerationJobConnection",
    "kind": "LinkedField",
    "name": "dataGenerationJobs",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "concreteType": "DataGenerationJobEdge",
        "kind": "LinkedField",
        "name": "edges",
        "plural": true,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "DataGenerationJob",
            "kind": "LinkedField",
            "name": "node",
            "plural": false,
            "selections": [
              (v1/*: any*/),
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "status",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "corpusSource",
                "storageKey": null
              },
              (v3/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "completedAt",
                "storageKey": null
              }
            ],
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ],
    "storageKey": "dataGenerationJobs(first:100)"
  },
  {
    "alias": null,
    "args": (v0/*: any*/),
    "concreteType": "LLMAdapterConnection",
    "kind": "LinkedField",
    "name": "llmAdapters",
    "plural": false,
    "selections": [
      {
        "alias": null,
        "args": null,
        "concreteType": "LLMAdapterEdge",
        "kind": "LinkedField",
        "name": "edges",
        "plural": true,
        "selections": [
          {
            "alias": null,
            "args": null,
            "concreteType": "LLMAdapter",
            "kind": "LinkedField",
            "name": "node",
            "plural": false,
            "selections": [
              (v1/*: any*/),
              (v2/*: any*/),
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "provider",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "modelName",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "canGenerate",
                "storageKey": null
              },
              {
                "alias": null,
                "args": null,
                "kind": "ScalarField",
                "name": "canJudge",
                "storageKey": null
              },
              (v3/*: any*/)
            ],
            "storageKey": null
          }
        ],
        "storageKey": null
      }
    ],
    "storageKey": "llmAdapters(first:100)"
  }
];
return {
  "fragment": {
    "argumentDefinitions": [],
    "kind": "Fragment",
    "metadata": null,
    "name": "DataGenerationPageQuery",
    "selections": (v4/*: any*/),
    "type": "Query",
    "abstractKey": null
  },
  "kind": "Request",
  "operation": {
    "argumentDefinitions": [],
    "kind": "Operation",
    "name": "DataGenerationPageQuery",
    "selections": (v4/*: any*/)
  },
  "params": {
    "cacheID": "b7df31591305a3db03a7ea4235e4d699",
    "id": null,
    "metadata": {},
    "name": "DataGenerationPageQuery",
    "operationKind": "query",
    "text": "query DataGenerationPageQuery {\n  dataGenerationJobs(first: 100) {\n    edges {\n      node {\n        id\n        name\n        status\n        corpusSource\n        createdAt\n        completedAt\n      }\n    }\n  }\n  llmAdapters(first: 100) {\n    edges {\n      node {\n        id\n        name\n        provider\n        modelName\n        canGenerate\n        canJudge\n        createdAt\n      }\n    }\n  }\n}\n"
  }
};
})();

(node as any).hash = "9a67cfa3c9d6a9c06110912e4e45059f";

export default node;
