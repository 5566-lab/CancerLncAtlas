"""Actual FastAPI routes for the current selected analysis and downloads."""
from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict, Field

from cc_hhgt.v32.current_analysis_web import CurrentAnalysisWebStore, CurrentWebQueryError, CurrentWebVersionError


class TableQuery(BaseModel):
    model_config=ConfigDict(extra="forbid",strict=True)
    version: str=Field(pattern=r"^[0-9a-f]{64}$")
    filters: dict[str,str]=Field(default_factory=dict)
    limit: int=Field(default=100,ge=1,le=1000)
    offset: int=Field(default=0,ge=0,le=1000000)


def current_analysis_router(store:CurrentAnalysisWebStore):
    router=APIRouter(prefix="/api/current",tags=["Current selected analysis"])

    def invoke(operation):
        try: return operation()
        except CurrentWebVersionError as error: raise HTTPException(409,detail=str(error)) from error
        except CurrentWebQueryError as error: raise HTTPException(422,detail=str(error)) from error
        except KeyError as error: raise HTTPException(404,detail="Unknown current table, part or unavailable component") from error
        except (ValueError,OSError) as error:
            # Native errors can contain filesystem paths. The public response
            # identifies an artifact failure without disclosing those paths.
            raise HTTPException(503,detail="Current release artifact verification failed") from error

    @router.get("/catalog")
    def catalog(): return invoke(store.catalog)

    @router.post("/tables/{identity}/query")
    def query(identity:str,request:TableQuery): return invoke(lambda:store.query(identity,**request.model_dump()))

    @router.get("/downloads/{identity}/{part}")
    def download(identity:str,part:int,version:str=Query(pattern=r"^[0-9a-f]{64}$")):
        record=invoke(lambda:store.download(identity,part,version=version))
        return FileResponse(record["path"],filename=record["filename"],media_type="application/octet-stream",
            headers={"X-Content-SHA256":record["sha256"],"X-Analysis-Version":record["version"],
                "X-Selected-Analysis-SHA256":record["selected_analysis_sha256"],"Cache-Control":"private, no-store"})

    @router.get("/drug-members")
    def drug_members(geneset_id:str,drug_id:str,version:str=Query(pattern=r"^[0-9a-f]{64}$"),
                     limit:int=Query(default=100,ge=1,le=1000),offset:int=Query(default=0,ge=0,le=1000000)):
        return invoke(lambda:store.drug_members(geneset_id=geneset_id,drug_id=drug_id,version=version,limit=limit,offset=offset))
    return router
